from Rockets.utils import parse_sf_params
from pathlib import Path



def from_screenshots(path, patt = "*"):
    if not isinstance(path, Path):
        path = Path(path)
    paths = list(path.glob(patt +".png"))
    files = [{"file":
                {"path": p.resolve().as_posix(), 
                 "name": p.name, 
                 "stem": p.stem},
              "SFparams": parse_sf_params(p.stem)
                } for p in paths] 
    return files


case1 = dict(
    m = int(4),
    a = 0.7,
    b = 0.7,
    n_1 = 2.585,
    n_2 = 9.281,
    n_3 = 9.281
)

case2 = dict(
    m = int(7),
    a = 0.7,
    b = 0.7,
    n_1 = 2.585,
    n_2 = 1.281,
    n_3 = 1.281
)

case3 = {'m': 5,
 'a': 0.725,
 'b': 0.725,
 'n_1': 7.223,
 'n_2': 13.575,
 'n_3': 13.575}


def from_consts():
    boop = [case1, case2, case3]
    boop = [{"SFparams": c, "file": "from_consts"} for c in boop]
    return boop