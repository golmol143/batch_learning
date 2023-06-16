
def frange(start, stop, step):
     x = start
     while x < stop and abs(stop - x) > 1e-6:
         yield x
         x += step

def hamming_weight(num: int) -> int:
        weight = 0

        while num:
            weight += 1
            num &= num - 1

        return weight
