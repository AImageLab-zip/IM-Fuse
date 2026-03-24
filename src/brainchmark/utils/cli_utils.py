import typer

def parse_crop_size(
    value: list[int] | None,
) -> tuple[int, int, int] | None:
    if value is None or len(value) == 0:
        return None
    if len(value) == 1:
        return value[0], value[0], value[0]
    if len(value) == 3:
        return value[0],value[1],value[2]
    raise typer.BadParameter(
        "Expected either 1 integer or 3 integers. "
        "Examples: `--crop-size 64` or `--crop-size 64 80 96`."
    )
def parse_percentile(
    value: list[float] | None,
) -> tuple[float,float] | None:
    if value is None or len(value) == 0:
        return None
    if len(value) == 1:
        return value[0], value[0], value[0]
    if len(value) == 3:
        return value[0],value[1],value[2]
    raise typer.BadParameter(
        "Expected either 1 integer or 3 integers. "
        "Examples: `--crop-size 64` or `--crop-size 64 80 96`."
    )
def parse_clamp(
    value: list[int] | None,
) -> tuple[int, int, int,int] | None:
    if value is None or len(value) == 0:
        return None
    if len(value) == 1:
        return value[0], value[0], value[0],value[0]
    if len(value) == 4:
        return value[0],value[1],value[2],value[3]
    raise typer.BadParameter(
        "Expected either 1 integer or 4 integers. "
        "Examples: `--clamp-min 0` or `--clamp-max 2000 3000 1000 5000`."
    )