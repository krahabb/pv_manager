# PV Manager
 Home assistant integration for photovoltaic systems management.

 This custom component contains a collection of helper features aimed at simplifying and enriching PV energy management in Home Assistant.

## WARNING
 > Since development is still in the early stages everything might be more or less **experimental**. Also, I'm planning to publish this component through the official HACS repository but at the moment you'd have to manually install it by following the typical manual installation procedure for custom components. (You could still use the HACS [custom repository](https://www.hacs.xyz/docs/faq/custom_repositories/) feature to simplify download and installation).
 Also, if you're *brave* enough to give it a try don't expect flawless behavior: comments are welcomed but a lot of development is still to be done both on published features and on new (awesome) ones so things can change a lot and compatibility between published releases might be 'fragile'. As stated, you should think of this as an *experiment*.

 Provided features:
 - **Energy meters**: These are classic 'Energy meters' for (pv) power sensors. They're not rocket science and you could already have implemented those by using official [(Riemann) Integral integration](https://www.home-assistant.io/integrations/integration/) maybe followed by an [Utility Meter](https://www.home-assistant.io/integrations/utility_meter/).
 The purpose here is to greatly simplify the configuration in a single compact entity and overcome some drawbacks of the standard approach which I've found a bit cumbersome.

 - **Energy forecast**: As the name implies, this is a component which forecasts energy production in the foreseeable future. Again, if you're a bit into it you might be using [Forecast.Solar](https://www.home-assistant.io/integrations/forecast_solar/) for the purpose with alternating fortune (at least in my experience). The key difference here is that prediction is mostly based on real data observed at your plant (in the field of estimation theory this can be regarded as a 'filter estimator') and not on the fixed (simple) model used by Forecast.Solar which bases it's predictions on your plant being exactly as configured (i.e. you provide peak-power, and orientation). Here, you don't use anything of this but just configure an 'observed' entity as input of data for which the component will provide future estimation. The model also needs some parameters to tune its behavior.
 At the moment it still doesn't use weather forecasts to improve prediction, nevertheless, for many weather patterns it still can prove correct since it actually uses recent observations during the day to tune its predictions. I'll provide a better detailed help on the estimator once it's done so at the moment you should think of it as a 'black-box' with some intuitive configuration parameters.

