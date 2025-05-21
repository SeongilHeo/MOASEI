# Wildfire Environment Configuration (WS#.pkl)

## Grid Setup
- Grid Dimensions: 3 x 3 (9 cells total)

## Fire Configuration
- Fire Types: (0,1,2)
    ``` python
    [[0, 2, 1],
    [2, 0, 2],
    [1, 2, 0]]
    ```
- Number of Fire States: 5
- Lit Cells: the fire is currently lit in each cell
    ``` python
    [[False, True, False],
    [False, False, False],
    [False, True, False]]
    ```
- Intensity Increase Probability: 0.9
- Intensity Decrease Probability: 0.85
- Extra Power Decrease Bonus: 0.15
- Burnout Probability: 0.5
- <span style="color: red;">Base Spread Rate: 0.1 (WS 1) / 0.5 (WS 2,3)</span>
- Max Spread Rate: 0.0 (limited spread)
- <span style="color: red;">Random Ignition Probability: 0.05 (WS 1) / 0.25 (WS 2,3)</span>
- Cell Size: 200.0
- Wind Direction: 0.0
- Ignition Temperature: 2 for all cells
- Initial Fuel: 2

## Agent Configuration
- Agent Positions:
    ``` python
    [[0, 0],
    [1, 1],
    [2, 2]]
    ```
- Fire Reduction Power: [1, 1, 1]
- Attack Range: [1, 1, 1]
- Suppressant States: 3
- <span style="color: red;">Initial Suppressant: 2 (WS 1,2) / 0 (WS 3)</span>
- Suppressant Decrease Probability: 1.0
- Suppressant Refill Probability: 1.0
- Initial Equipment State: 2
- Equipment States: 3 x 3 matrix of zeros
- Repair Probability: 1.0
- Degrade Probability: 1.0
- Critical Error Probability: 0.0
- Initial Capacity: 2
- Tank Switch Probability: 1.0
- Possible Capacities: [1.0, 2.0, 3.0]
- Capacity Probabilities: [0.0, 1.0, 0.0] (always 2)

## Reward Configuration
- Fire Rewards Matrix:
    ``` python
    [[0, 4, 2],
     [4, 0, 4],
     [2, 4, 0]]
    ```
- Bad Attack Penalty: -100.0
- Burnout Penalty: -1.0
- Termination Reward: 0.0

## Stochastic Configuration
- Special Burnout Probability: <span style="color: cyan;">Enabled</span>
- Suppressant Refill: <span style="color: cyan;">Enabled</span>
- Suppressant Decrease: <span style="color: cyan;">Enabled</span>
- Tank Switch: <span style="color: magenta;">Disabled</span>
- Critical Error: <span style="color: magenta;">Disabled</span>
- Degrade: <span style="color: magenta;">Disabled</span>
- Repair: <span style="color: magenta;">Disabled</span>
- Fire Increase: <span style="color: cyan;">Enabled</span>
- Fire Decrease: <span style="color: cyan;">Enabled</span>
- Fire Spread: <span style="color: cyan;">Enabled</span>
- Realistic Fire Spread: <span style="color: magenta;">Disabled</span>
- Random Fire Ignition: <span style="color: cyan;">Enabled</span>
- Fire Fuel: <span style="color: magenta;">Disabled</span>


# Evaluation

- the total number of fires extinguished vs. burned out
- the average duration of each fire
- the efficiency of limited suppressant usage by agents


# Penalty

- In_range
- Refilling


* 1 2
1 1 2
2 2 2

1 1 1
1 * 1
1 1 1


2 2 2
2 1 1
2 1 *