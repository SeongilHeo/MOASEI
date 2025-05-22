from free_range_zoo.envs import wildfire_v0

all_prefixes = ["oasys_mas"]

# environments which have manual policy scripts, allowing interactive play
manual_environments = {}

oasys_mas = {
    'oasys_mas/wildfire_v0': wildfire_v0,
}

all_environments = {
    **oasys_mas,
}
