import gym
import numpy as np
import pandas as pd


class DailySimulation(gym.Env):
    """
    The observation is a dictionary that has as key the name of the product and as value
    the OnOrder divided by residual lead time and the inventory divided by Residual shelf life.

    The action is assumed as a np.array that preserves the order of the prod_setting dictionary.
    Each element is the number of items per product to be ordered that day.
    """

    def __init__(self, scenarioMgr, timeHorizon, invManagers, supManagers, statMgr, consumer, flagPrint=False):
        super(DailySimulation, self).__init__()

        self.scenarioMgr = scenarioMgr
        self.timeHorizon = timeHorizon
        self.invManagers = invManagers
        self.supManagers = supManagers
        self.statMgr = statMgr
        self.statMgr.setTimeHorizon(self.timeHorizon)
        self.consumer = consumer
        self.flagPrint = flagPrint

        self.current_step = 0
        self.scenario = self.scenarioMgr.makeScenario(self.timeHorizon)

        self.history = {}
        self.sales = {}

        # dataset logging
        self.dataset_rows = []

        for k in self.invManagers.keys():
            self.history[k] = []
            self.sales[k] = np.zeros(self.invManagers.get(k).shelfLife)

    def _build_observation(self):
        obs = {}
        for k in self.invManagers.keys():
            obs[k] = np.concatenate(
                (
                    np.flip(self.supManagers.get(k).OnOrder[:-1]),
                    np.flip(self.invManagers.get(k).inventory[:-1]),
                ),
                axis=0,
            )
        obs["Day"] = self.current_step % 7
        return obs

    def _log_step_data(
        self,
        day_idx,
        pre_inventory,
        pre_on_order,
        action,
        scrapped,
        lost_clients,
        unmet_clients,
        reward,
        done,
    ):
        for i, k in enumerate(self.invManagers.keys()):
            row = {
                "day_idx": int(day_idx),
                "weekday": int(day_idx % 7),
                "product": k,
                "action_ordered": float(action[i]),
                "total_sold": float(np.sum(self.sales[k])),
                "scrapped": float(scrapped[i]),
                "lost_clients": int(lost_clients),
                "unmet_clients": int(unmet_clients),
                "reward": float(reward),
                "done": bool(done),
            }

            # pre-decision state
            for j, val in enumerate(pre_inventory[k]):
                row[f"inv_age_slot_{j}"] = float(val)

            for j, val in enumerate(pre_on_order[k]):
                row[f"pipeline_slot_{j}"] = float(val)

            # post-outcome sales by age
            for j, val in enumerate(self.sales[k]):
                row[f"sold_age_slot_{j}"] = float(val)

            self.dataset_rows.append(row)

    def get_dataset(self):
        return pd.DataFrame(self.dataset_rows)

    def save_dataset_csv(self, filepath="./simulation_dataset.csv"):
        df = self.get_dataset()
        df.to_csv(filepath, index=False)

    def reset(self):
        self.current_step = 0
        obs = {}

        for k in self.invManagers.keys():
            self.invManagers.get(k).clearState()
            self.supManagers.get(k).clearState()
            self.history[k] = []
            self.sales[k] = np.zeros(self.invManagers.get(k).shelfLife)

            obs[k] = np.concatenate(
                (
                    np.flip(self.supManagers.get(k).OnOrder[:-1]),
                    np.flip(self.invManagers.get(k).inventory[:-1]),
                ),
                axis=0,
            )

        obs["Day"] = 0
        self.statMgr.clearStatistics()
        self.scenario = self.scenarioMgr.makeScenario(self.timeHorizon)
        self.dataset_rows = []
        return obs

    def step(self, action: np.array):
        # robust termination guard
        if self.current_step >= self.timeHorizon:
            raise RuntimeError("step() called after the simulation already ended. Call reset().")

        day_idx = self.current_step

        # capture pre-decision state for dataset
        pre_inventory = {
            k: self.invManagers.get(k).inventory.copy() for k in self.invManagers.keys()
        }
        pre_on_order = {
            k: self.supManagers.get(k).OnOrder.copy() for k in self.supManagers.keys()
        }

        # place orders
        for i, k in enumerate(self.supManagers.keys()):
            self.supManagers.get(k).GetOrder(action[i])
            self.history.get(k).append(action[i])
            self.sales[k] = np.zeros(self.invManagers.get(k).shelfLife)

        self.statMgr.updateClock()

        if self.flagPrint:
            print("------------------------------------------")
            print("\n day", day_idx, "\n inventory:")
            for k in self.invManagers.keys():
                print("Product", k, ": Stored")
                for i in range(self.invManagers.get(k).shelfLife - 1):
                    print(
                        "\t",
                        self.invManagers.get(k).inventory[i],
                        "items with",
                        i + 1,
                        "Residual shelf life",
                    )

            print(" onOrder:")
            for k in self.supManagers.keys():
                print("Product", k, ":Waiting for")
                for i in range(self.supManagers.get(k).LeadTime + 1):
                    if i != 0:
                        print(
                            "\t",
                            np.ceil(self.supManagers.get(k).OnOrder[i]),
                            "items, expected in",
                            i,
                            "days",
                        )
                    else:
                        print(
                            "\t",
                            np.ceil(self.supManagers.get(k).OnOrder[i]),
                            "items have just arrived.",
                        )

            print("Total demand (partially lost): ", self.scenario[0][day_idx])

        # receive deliveries
        for k in self.invManagers.keys():
            delivered = self.supManagers.get(k).deliverSupply()
            self.invManagers.get(k).receiveSupply(delivered)

        lostClients = 0
        unmetClients = 0

        # simulate customers for this day
        todays_demand = int(np.floor(self.scenario[0][day_idx]))
        for _ in range(todays_demand):
            availability = []
            for k in self.invManagers.keys():
                availability.extend(self.invManagers.get(k).getProductAvailabilty())

            if any(availability):
                choice = self.consumer.makeChoice(availability)

                if choice == -1:
                    lostClients += 1
                else:
                    productKey = self.statMgr.keys_list_age[choice]
                    inventoryMgr = self.invManagers[productKey]
                    ageArray = inventoryMgr.shelfLife - np.cumsum(
                        self.statMgr.keys_list_age == productKey
                    )
                    sold = inventoryMgr.meetDemand(ageArray[choice])
                    self.sales[productKey][inventoryMgr.shelfLife - ageArray[choice] - 1] += sold
            else:
                unmetClients += 1

        # end of day aging / scrapping
        scrapped = np.zeros(self.statMgr.nProducts)
        for i, k in enumerate(self.invManagers.keys()):
            scrapped[i] = self.invManagers.get(k).updateInventory()

        self.statMgr.updateUnmet(unmetClients)
        self.statMgr.updateLost(lostClients)
        reward = self.statMgr.updateStats(action, self.sales, scrapped)

        # advance clock AFTER processing the day
        self.current_step += 1
        done = self.current_step >= self.timeHorizon

        if self.flagPrint:
            for i in range(action.size):
                productKey = self.statMgr.keys_list[i]
                print(
                    "Product ",
                    productKey,
                    " Ordered: ",
                    action[i],
                    " Sold:  ",
                    self.sales.get(productKey),
                    " Scrapped: ",
                    scrapped[i],
                )
            print(" No purchase: ", lostClients, "Unmet Demand: ", unmetClients)
            print("Total unmet so far", self.statMgr.TotalUnmetDemand)
            print("Total ordered so far ", sum(self.statMgr.TotalOrdered))
            print("Total scrapped so far", sum(self.statMgr.TotalScrapped))
            print("Total sold so far", self.statMgr.TotalSold)
            print("Profit of the day ", reward)

        # log dataset row(s)
        self._log_step_data(
            day_idx=day_idx,
            pre_inventory=pre_inventory,
            pre_on_order=pre_on_order,
            action=action,
            scrapped=scrapped,
            lost_clients=lostClients,
            unmet_clients=unmetClients,
            reward=reward,
            done=done,
        )

        obs = self._build_observation()

        if self.flagPrint:
            print("State observation: ", obs)

        if done and self.flagPrint:
            print(
                "Simulation metrics:\n\tAverage Profit = ",
                self.getAverageProfit(),
                "\n\tAverage Waste = ",
                self.getAverageScrapped(),
            )

        return obs, reward, done, {}

    def setSeed(self, seed):
        self.scenarioMgr.setSeed(seed)

    def updateHorizon(self, timeHorizon):
        self.timeHorizon = timeHorizon
        self.statMgr.setTimeHorizon(self.timeHorizon)

    def getAverageProfit(self):
        return self.statMgr.getAverageProfit()

    def getAverageScrapped(self):
        return self.statMgr.getAverageScrapped()

    def getLostClients(self):
        return self.statMgr.TotalLostDemand

    def getUnmetClients(self):
        return self.statMgr.TotalUnmetDemand