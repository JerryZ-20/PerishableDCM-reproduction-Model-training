import numpy as np


class InventoryManager:
    """
    Inventory manager for one product with probabilistic aging.

    Inventory is stored in age buckets from oldest to freshest:
        inventory[0]                -> oldest items, expire next aging event
        inventory[shelfLife - 1]    -> freshest items

    At the end of each day, each unit can:
        - age by 0 buckets with probability p0
        - age by 1 bucket with probability p1
        - age by 2 buckets with probability p2

    Here:
        p0 and p2 are sampled fresh each day from a binomial-based proportion
        with expected value about `aging_prob_mean`.
        p1 = 1 - p0 - p2

    Any items pushed past bucket 0 are scrapped.
    """

    def __init__(self, shelfLife: int, aging_prob_mean: float = 0.15, aging_binom_n: int = 20):
        if shelfLife <= 0:
            raise ValueError("shelfLife must be positive")
        if not (0 <= aging_prob_mean <= 0.5):
            raise ValueError("aging_prob_mean must be between 0 and 0.5")
        if aging_binom_n <= 0:
            raise ValueError("aging_binom_n must be positive")

        self.shelfLife = shelfLife
        self.inventory = np.zeros(shelfLife, dtype=int)

        # Controls the random daily probabilities for aging by 0 and aging by 2.
        # Example: Binomial(n=20, p=0.15) / 20 has mean 0.15.
        self.aging_prob_mean = aging_prob_mean
        self.aging_binom_n = aging_binom_n

        # Stores the most recently sampled probabilities, useful for debugging/logging.
        self.last_p0 = 0.0
        self.last_p1 = 1.0
        self.last_p2 = 0.0

    def clearState(self):
        self.inventory = np.zeros(self.shelfLife, dtype=int)

    def _sample_daily_aging_probs(self):
        """
        Sample p0 and p2 from binomial-based proportions with mean aging_prob_mean.
        Resample until p0 + p2 <= 1, then set p1 as the remainder.
        """
        while True:
            p0 = np.random.binomial(self.aging_binom_n, self.aging_prob_mean) / self.aging_binom_n
            p2 = np.random.binomial(self.aging_binom_n, self.aging_prob_mean) / self.aging_binom_n
            if p0 + p2 <= 1.0:
                break

        p1 = 1.0 - p0 - p2

        self.last_p0 = p0
        self.last_p1 = p1
        self.last_p2 = p2

        return p0, p1, p2

    def updateInventory(self):
        """
        Apply one day of probabilistic aging to all inventory.

        Returns
        -------
        scrapped : int
            Number of items that expired during this update.
        """
        p0, p1, p2 = self._sample_daily_aging_probs()

        old_inventory = self.inventory.copy()
        new_inventory = np.zeros(self.shelfLife, dtype=int)
        scrapped = 0

        # Process each bucket independently.
        # Bucket i means: i=0 oldest, i=shelfLife-1 freshest.
        for i in range(self.shelfLife):
            count = int(old_inventory[i])
            if count == 0:
                continue

            stay_count, age1_count, age2_count = np.random.multinomial(count, [p0, p1, p2])

            # Age by 0: stay in same bucket
            new_inventory[i] += stay_count

            # Age by 1: move one bucket older
            if i - 1 >= 0:
                new_inventory[i - 1] += age1_count
            else:
                scrapped += age1_count

            # Age by 2: move two buckets older
            if i - 2 >= 0:
                new_inventory[i - 2] += age2_count
            else:
                scrapped += age2_count

        self.inventory = new_inventory
        return scrapped

    def receiveSupply(self, orderSize):
        self.inventory[self.shelfLife - 1] += int(np.floor(orderSize))

    def meetDemand(self, age):
        if (not self.isAvailable()) or (not self.isAvailableAge(age)):
            raise ValueError("The customer cannot buy something missing")

        sales = 1
        self.inventory[self.shelfLife - age - 1] -= sales
        return sales

    def isAvailable(self):
        return np.any(self.inventory > 0)

    def isAvailableAge(self, age):
        if age >= self.shelfLife or age < 0:
            raise ValueError("Age out of the bounds for this product")
        return self.inventory[self.shelfLife - age - 1] >= 1

    def getProductAvailabilty(self):
        return list(map(bool, self.inventory.tolist()))