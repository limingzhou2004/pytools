class MlFlowConfig:
    def __init__(self, address, model_config, modeling_setting, weather_para):
        self.model_config = model_config
        self.model_setting = modeling_setting
        self.address = address
        self.weather_para = weather_para
