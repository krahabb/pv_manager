"""
HA history helpers:
trying to optimize the query for history data for our needs
"""

from datetime import datetime
from typing import TYPE_CHECKING

from homeassistant import const as hac
from homeassistant.components.recorder.history.modern import (
    _FIELD_MAP,
    COMPRESSED_STATE_LAST_UPDATED,
    COMPRESSED_STATE_STATE,
    LAST_CHANGED_KEY,
    NEED_ATTRIBUTE_DOMAINS,
    SHARED_ATTR_OR_LEGACY_ATTRIBUTES,
    SIGNIFICANT_DOMAINS,
    STATE_KEY,
    LazyState,
    StateAttributes,
    States,
    execute_stmt_lambda_element,
    extract_metadata_ids,
    get_instance,
    groupby,
    itemgetter,
    lambda_stmt,
    row_to_compressed_state,
    select,
    session_scope,
    split_entity_id,
)
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        Final,
        Iterable,
        Iterator,
        Mapping,
        NotRequired,
        TypedDict,
    )

    from homeassistant.components.recorder.history.modern import (
        CompoundSelect,
        Row,
        Select,
        Session,
        StatementLambdaElement,
        Subquery,
    )
    from homeassistant.core import HomeAssistant, State

    class CompressedState(TypedDict):
        """Compressed dict of a state.
        HA core class seems bugged but I have to investigate
        TODO: check and PR to fix it.
        It seems core.State.as_compressed_state sets 'changed' and optionally 'updated'
        while history state changed queries do the opposite !?
        In our history query anyway lu is always returned.
        """

        s: str  # COMPRESSED_STATE_STATE
        a: Mapping[str, Any]  # COMPRESSED_STATE_ATTRIBUTES
        c: str | dict[str, Any]  # COMPRESSED_STATE_CONTEXT
        lc: NotRequired[float]  # COMPRESSED_STATE_LAST_CHANGED
        lu: float  # COMPRESSED_STATE_LAST_UPDATED


def get_significant_states(
    hass: "HomeAssistant",
    entity_ids: list[str],
    start_time_ts: float,
    end_time_ts: float | None = None,
    significant_changes_only: bool = True,
    minimal_response: bool = False,
    no_attributes: bool = False,
    compressed_state_format: bool = False,
) -> "dict[str, list[State | dict[str, Any]]]":
    """Wrap get_significant_states_with_session with an sql session."""
    with session_scope(hass=hass, read_only=True) as session:
        return get_significant_states_with_session(
            hass,
            session,
            entity_ids,
            start_time_ts,
            end_time_ts,
            significant_changes_only,
            minimal_response,
            no_attributes,
            compressed_state_format,
        )


def get_significant_states_with_session(
    hass: "HomeAssistant",
    session: "Session",
    entity_ids: list[str],
    start_time_ts: float,
    end_time_ts: float | None = None,
    significant_changes_only: bool = True,
    minimal_response: bool = False,
    no_attributes: bool = False,
    compressed_state_format: bool = False,
) -> "dict[str, list[State | dict[str, Any]]]":
    """Return states changes during UTC period start_time - end_time.

    entity_ids is an optional iterable of entities to include in the results.

    filters is an optional SQLAlchemy filter which will be applied to the database
    queries unless entity_ids is given, in which case its ignored.

    Significant states are all states where there is a state change,
    as well as all states from certain domains (for instance
    thermostat so that we get current temperature in our graphs).
    """
    entity_id_to_metadata_id: dict[str, int | None] | None = None
    metadata_ids_in_significant_domains: list[int] = []
    instance = get_instance(hass)
    if not (
        entity_id_to_metadata_id := instance.states_meta_manager.get_many(
            entity_ids, session, False
        )
    ) or not (possible_metadata_ids := extract_metadata_ids(entity_id_to_metadata_id)):
        return {}
    metadata_ids = possible_metadata_ids
    if significant_changes_only:
        metadata_ids_in_significant_domains = [
            metadata_id
            for entity_id, metadata_id in entity_id_to_metadata_id.items()
            if metadata_id is not None
            and split_entity_id(entity_id)[0] in SIGNIFICANT_DOMAINS
        ]

    stmt = _generate_significant_states_with_session_stmt(
        start_time_ts,
        end_time_ts,
        metadata_ids,
        metadata_ids_in_significant_domains,
        significant_changes_only,
        no_attributes,
    )
    rows: "list[Row]" = execute_stmt_lambda_element(session, stmt, None, None, orm_rows=False)  # type: ignore
    return _sorted_states_to_dict(
        rows,
        entity_ids,
        entity_id_to_metadata_id,
        minimal_response,
        compressed_state_format,
        no_attributes=no_attributes,
    )


def _generate_significant_states_with_session_stmt(
    start_time_ts: float,
    end_time_ts: float | None,
    metadata_ids: list[int],
    metadata_ids_in_significant_domains: list[int],
    significant_changes_only: bool,
    no_attributes: bool,
) -> "StatementLambdaElement":
    return lambda_stmt(
        lambda: _significant_states_stmt(
            start_time_ts,
            end_time_ts,
            metadata_ids,
            metadata_ids_in_significant_domains,
            significant_changes_only,
            no_attributes,
        ),
        track_on=[
            bool(metadata_ids_in_significant_domains),
            bool(end_time_ts),
            significant_changes_only,
            no_attributes,
        ],
    )


def _significant_states_stmt(
    start_time_ts: float,
    end_time_ts: float | None,
    metadata_ids: list[int],
    metadata_ids_in_significant_domains: list[int],
    significant_changes_only: bool,
    no_attributes: bool,
) -> "Select | CompoundSelect":
    """Query the database for significant state changes."""
    include_last_changed = not significant_changes_only
    stmt = _stmt_and_join_attributes(no_attributes, include_last_changed, False)
    if significant_changes_only:
        # Since we are filtering on entity_id (metadata_id) we can avoid
        # the join of the states_meta table since we already know which
        # metadata_ids are in the significant domains.
        if metadata_ids_in_significant_domains:
            stmt = stmt.filter(
                States.metadata_id.in_(metadata_ids_in_significant_domains)
                | (States.last_changed_ts == States.last_updated_ts)
                | States.last_changed_ts.is_(None)
            )
        else:
            stmt = stmt.filter(
                (States.last_changed_ts == States.last_updated_ts)
                | States.last_changed_ts.is_(None)
            )
    stmt = stmt.filter(States.metadata_id.in_(metadata_ids)).filter(
        States.last_updated_ts > start_time_ts
    )
    if end_time_ts:
        stmt = stmt.filter(States.last_updated_ts < end_time_ts)
    if not no_attributes:
        stmt = stmt.outerjoin(
            StateAttributes, States.attributes_id == StateAttributes.attributes_id
        )
    return stmt.order_by(States.metadata_id, States.last_updated_ts)


def _stmt_and_join_attributes(
    no_attributes: bool,
    include_last_changed: bool,
    include_last_reported: bool,
) -> "Select":
    """Return the statement and if StateAttributes should be joined."""
    _select = select(States.metadata_id, States.state, States.last_updated_ts)
    if include_last_changed:
        _select = _select.add_columns(States.last_changed_ts)
    if include_last_reported:
        _select = _select.add_columns(States.last_reported_ts)
    if not no_attributes:
        _select = _select.add_columns(SHARED_ATTR_OR_LEGACY_ATTRIBUTES)
    return _select


def _sorted_states_to_dict(
    states: "Iterable[Row]",
    entity_ids: list[str],
    entity_id_to_metadata_id: dict[str, int | None],
    minimal_response: bool = False,
    compressed_state_format: bool = False,
    no_attributes: bool = False,
) -> "dict[str, list[State | dict[str, Any]]]":
    """Convert SQL results into JSON friendly data structure.

    This takes our state list and turns it into a JSON friendly data
    structure {'entity_id': [list of states], 'entity_id2': [list of states]}

    States must be sorted by entity_id and last_updated

    We also need to go back and create a synthetic zero data point for
    each list of states, otherwise our graphs won't start on the Y
    axis correctly.
    """
    field_map = _FIELD_MAP
    state_class: "Callable[[Row, dict[str, dict[str, Any]], float | None, str, str, float | None, bool],State | dict[str, Any]]"
    if compressed_state_format:
        state_class = row_to_compressed_state
        attr_time = COMPRESSED_STATE_LAST_UPDATED
        attr_state = COMPRESSED_STATE_STATE
    else:
        state_class = LazyState
        attr_time = LAST_CHANGED_KEY
        attr_state = STATE_KEY

    # Set all entity IDs to empty lists in result set to maintain the order
    result: "dict[str, list[State | dict[str, Any]]]" = {
        entity_id: [] for entity_id in entity_ids
    }
    metadata_id_to_entity_id: dict[int, str] = {}
    metadata_id_to_entity_id = {
        v: k for k, v in entity_id_to_metadata_id.items() if v is not None
    }
    # Get the states at the start time
    if len(entity_ids) == 1:
        metadata_id = entity_id_to_metadata_id[entity_ids[0]]
        assert metadata_id is not None  # should not be possible if we got here
        states_iter: "Iterable[tuple[int, Iterator[Row]]]" = (
            (metadata_id, iter(states)),
        )
    else:
        key_func = itemgetter(field_map["metadata_id"])
        states_iter = groupby(states, key_func)

    state_idx = field_map["state"]
    last_updated_ts_idx = field_map["last_updated_ts"]

    # Append all changes to it
    for metadata_id, group in states_iter:
        entity_id = metadata_id_to_entity_id[metadata_id]
        attr_cache: dict[str, dict[str, Any]] = {}
        ent_results = result[entity_id]
        if (
            not minimal_response
            or split_entity_id(entity_id)[0] in NEED_ATTRIBUTE_DOMAINS
        ):
            ent_results.extend(
                [
                    state_class(
                        db_state,
                        attr_cache,
                        None,
                        entity_id,
                        db_state[state_idx],
                        db_state[last_updated_ts_idx],
                        False,
                    )
                    for db_state in group
                ]
            )
            continue

        prev_state: str | None = None
        # With minimal response we only provide a native
        # State for the first and last response. All the states
        # in-between only provide the "state" and the
        # "last_changed".
        if not ent_results:
            if (first_state := next(group, None)) is None:
                continue
            prev_state = first_state[state_idx]
            ent_results.append(
                state_class(
                    first_state,
                    attr_cache,
                    None,
                    entity_id,
                    prev_state,  # type: ignore
                    first_state[last_updated_ts_idx],
                    no_attributes,
                )
            )

        #
        # minimal_response only makes sense with last_updated == last_updated
        #
        # We use last_updated for for last_changed since its the same
        #
        # With minimal response we do not care about attribute
        # changes so we can filter out duplicate states
        if compressed_state_format:
            # Compressed state format uses the timestamp directly
            ent_results.extend(
                [
                    {
                        attr_state: (prev_state := state),
                        attr_time: row[last_updated_ts_idx],
                    }
                    for row in group
                    if (state := row[state_idx]) != prev_state
                ]
            )
            continue

        # Non-compressed state format returns an ISO formatted string
        _utc_from_timestamp = dt_util.utc_from_timestamp
        ent_results.extend(
            [
                {
                    attr_state: (prev_state := state),
                    attr_time: _utc_from_timestamp(
                        row[last_updated_ts_idx]
                    ).isoformat(),
                }
                for row in group
                if (state := row[state_idx]) != prev_state
            ]
        )

    # Filter out the empty lists if some states had 0 results.
    return {key: val for key, val in result.items() if val}
