def slidingWindow(sequence, winSize, step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise ValueError("sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise ValueError("type(winSize) and type(step) must be int.")
    if step > winSize:
        raise ValueError("step must not be larger than winSize.")
    if winSize > len(sequence):
        raise ValueError("winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize) // step) + 1

    # Do the work
    for i in range(0, numOfChunks * step, step):
        yield sequence[i:i + winSize]


from itertools import islice


def sliding_window(seq, n=4):
    if len(seq) < n:
        yield seq
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
