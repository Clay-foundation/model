# sys.path.append("./")
from datacube import main
from tile import tiler


def run_stack_tile():
    stack = main()
    tiles = tiler(stack)
    print("Stack: ", stack)
    return tiles


tiles = run_stack_tile()
print("Number of tiles generated: ", len(tiles))
