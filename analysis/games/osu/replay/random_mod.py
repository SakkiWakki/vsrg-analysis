"""Port of .NET System.Random (osu!stable's RNG) + Random-mod
column shuffle, matching ManiaModRandom.ApplyToBeatmap."""


class _LegacyRandom:
    """Port of .NET System.Random (osu!stable's RNG). Ref: osu!framework
    LegacyRandom.cs — subtractive generator with seed array of 56 ints."""
    MBIG = 2147483647
    MSEED = 161803398

    def __init__(self, seed):
        ii = 0
        mj = self.MSEED - abs(int(seed))
        seed_arr = [0] * 56
        seed_arr[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            seed_arr[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += self.MBIG
            mj = seed_arr[ii]
        for _ in range(1, 5):
            for i in range(1, 56):
                seed_arr[i] -= seed_arr[1 + (i + 30) % 55]
                if seed_arr[i] < 0:
                    seed_arr[i] += self.MBIG
        self._seed_arr = seed_arr
        self._inext = 0
        self._inextp = 21

    def _sample(self):
        ni = self._inext + 1
        if ni >= 56:
            ni = 1
        np_ = self._inextp + 1
        if np_ >= 56:
            np_ = 1
        ret = self._seed_arr[ni] - self._seed_arr[np_]
        if ret == self.MBIG:
            ret -= 1
        if ret < 0:
            ret += self.MBIG
        self._seed_arr[ni] = ret
        self._inext = ni
        self._inextp = np_
        return ret

    def next_int(self, max_value):
        """Return integer in [0, max_value)."""
        return int(self._sample() * (1.0 / self.MBIG) * max_value)


def _random_column_permutation(keycount, seed):
    """osu!mania Random-mod column shuffle. Fisher-Yates using LegacyRandom,
    matching osu!stable's ManiaModRandom.ApplyToBeatmap shuffle logic."""
    rng = _LegacyRandom(seed)
    perm = list(range(keycount))
    for i in range(keycount - 1, 0, -1):
        j = rng.next_int(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm
