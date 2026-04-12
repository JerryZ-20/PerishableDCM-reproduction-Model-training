import numpy as np

class ScenarioGeneratorRandom:
    def __init__(self, store_setting: dict):
        seasonalityRaw = np.array(store_setting['Seasonals'])
        self.seasonality = seasonalityRaw / seasonalityRaw.mean()
        self.mu = self.seasonality * store_setting['ev_Daily']
        self.distr = store_setting['Distr']

        if self.distr == 'Normal':
            self.sigma = self.seasonality * store_setting['std_Daily']

    def reset(self):
        np.random.seed(None)

    def setSeed(self, seed):
        np.random.seed(seed)

    def makeScenario(self, timeHorizon):
        self.checkTimeHorizon(timeHorizon)

        if self.distr == 'Normal':
            scenario = np.maximum(
                0,
                np.random.normal(
                    self.mu[np.arange(timeHorizon) % 7],
                    self.sigma[np.arange(timeHorizon) % 7]
                )
            )
        elif self.distr == 'Poisson':
            scenario = np.random.poisson(self.mu[np.arange(timeHorizon) % 7])
        else:
            raise ValueError("Distribution not supported")

        return scenario.reshape(1, -1)

    @staticmethod
    def checkTimeHorizon(timeHorizon: int):
        if timeHorizon % 7:
            raise ValueError('The environment is weekly based. TimeHorizon must be multiple of 7')
