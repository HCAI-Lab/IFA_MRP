tasks = {
    "exploration": {
        1: {
            "subtask": "Explore the position 'fridge'.",
            "trajectories": [
                "go to fridge"
            ]
        },

        2: {
            "subtask": "Explore the position 'table 1'.",
            "trajectories": [
                "go to table 1"
            ]
        },

        3: {
            "subtask": "Explore the position 'sink'.",
            "trajectories": [
                "go to sink"
            ]
        },

        4: {
            "subtask": "Explore the position 'microwave'.",
            "trajectories": [
                "go to microwave"
            ]
        },

        5: {
            "subtask": "Explore the position 'stove'.",
            "trajectories": [
                "go to stove"
            ]
        },

        6: {
            "subtask": "Explore the position 'table 2'.",
            "trajectories": [
                "go to table 2"
            ]
        },

        7: {
            "subtask": "Explore the position 'door'.",
            "trajectories": [
                "go to door"
            ]
        },
    },
    "rearrangement": {
        1:{
            'subtask': "Place ‘tomato’ inside the microwave.", 
            'trajectories': 
               [
                   "go to table 1", 
                    "take tomato 1 from countertop 2", 
                    "go to microwave", 
                    "open microwave", 
                    "put tomato 1 in/on microwave 1", 
                    "close microwave",             
                ]
            },
        2:{
            'subtask': "Place ‘egg’ inside the fridge.", 
            'trajectories':         
                [
                    "go to table 1", 
                    "take egg 1 from countertop 2",
                    "go to fridge",
                    "open fridge 1", 
                    "put egg 1 in/on fridge 1", 
                    "close fridge 1",
                ],
            },
        3:{
            'subtask': "Place ‘spatula’ inside the top-right drawer below the stove.", 
            'trajectories':         
                [
                    "go to stove", 
                    "take spatula 1 from countertop 1",
                    "open drawer 3", 
                    "put spatula 1 in/on drawer 3", 
                    "close drawer 3",
                ],
            },
        4:{
            'subtask': "Place ‘kettle’ inside the cabinet above the coffee machine.", 
            'trajectories':         
                [
                    "take kettle 1 from countertop 1",
                    "go to microwave",
                    "open cabinet 8", 
                    "put kettle 1 in/on cabinet 8", 
                    "close cabinet 8",
                ],
            },
        5:{
            'subtask': "Place ‘pot’ on any of the stove burners.", 
            'trajectories':         
                [
                    "take pot 1 from countertop 1",
                    "go to stove",
                    "put pot 1 in/on stoveburner 2", 
                ],
            },
        6:{
            'subtask': "Place ‘spoon’ inside the bottom-right drawer below the stove.", 
            'trajectories':         
                [
                    "take spoon 1 from countertop 1",
                    "open drawer 2", 
                    "put spoon 1 in/on drawer 2", 
                    "close drawer 2",
                ],
            },
        7:{
            'subtask': "Place ‘ladle’ inside the drawer below the countertop next to the sink.", 
            'trajectories':         
                [
                    "take ladle 1 from countertop 1",
                    "go to sink",
                    "open drawer 12", 
                    "put ladle 1 in/on drawer 12", 
                    "close drawer 12",
                ],
            },
        8:{
            'subtask': "Place ‘soap bottle’ inside the cabinet above the sink.", 
            'trajectories':         
                [
                    "take soapbottle 1 from countertop 1",
                    "open cabinet 3", 
                    "put soapbottle 1 in/on cabinet 3", 
                    "close cabinet 3",
                ],
            },
        9:{
            'subtask': "Place ‘dish sponge’ inside the top-left drawer below the stove.", 
            'trajectories':         
                [
                    "take dishsponge 1 from countertop 1",
                    "go to stove",
                    "open drawer 7", 
                    "put dishsponge 1 in/on drawer 7", 
                    "close drawer 7",
                ],
            },
        10:{
            'subtask': "Place ‘fork’ inside the second drawer from the top at the center below the stove.", 
            'trajectories':         
                [
                    "take fork 1 from countertop 1",
                    "open drawer 9", 
                    "put fork 1 in/on drawer 9", 
                    "close drawer 9",
                ],
            },
        11:{
            'subtask': "Place ‘bowl’ inside the cabinet below the coffee machine.", 
            'trajectories':         
                [
                    "go to table 2",
                    "take bowl 1 from countertop 2",
                    "go to microwave",
                    "open cabinet 9", 
                    "put bowl 1 in/on cabinet 9", 
                    "close cabinet 9",
                ],
            },
        12:{
            'subtask': "Place ‘knife’ inside the second drawer from the bottom at the center below the stove.", 
            'trajectories':         
                [
                    "take knife 1 from countertop 1",
                    "go to stove",
                    "open drawer 8", 
                    "put knife 1 in/on drawer 8", 
                    "close drawer 8",
                ],
            },    
        13:{
            'subtask': "Find ‘cup’ from the cabinet above the coffee machine and place it inside the left-most cabinet below the sink.", 
            'trajectories':         
                [
                    "go to microwave",
                    "open cabinet 8",
                    "take cup 1 from cabinet 8",
                    "close cabinet 8",
                    "go to sink", 
                    "open cabinet 6",
                    "put cup 1 in/on cabinet 6", 
                    "close cabinet 6",
                ],
            },
        14:{
            'subtask': "Place ‘pan’ on any of the stove burners.", 
            'trajectories':         
                [
                    "take pan 1 from countertop 1",
                    "go to stove", 
                    "put pan 1 in/on stoveburner 6"
                ],
            },
        15:{
            'subtask': "Move to the position 'Door'.", 
            'trajectories':         
                [
                    "go to door"
                ],
            },    
    }, 
    "findNplace" : {
        1:{
            'subtask': "Find ‘apple’ and move it to the sinkbasin.", 
            'trajectories': 
               [
                   "go to fridge", "open fridge 1", "take apple 1 from fridge 1", "close fridge 1", 
                   "go to sink", "put apple 1 in/on sinkbasin 1"
                ]
            },
        2:{
            'subtask': "Find ‘butter knife’ and move it to the sinkbasin.", 
            'trajectories':         
                [
                   "go to stove", "open drawer 3", "take butterknife 1 from drawer 3", "close drawer 3", 
                   "go to sink", "put butterknife 1 in/on sinkbasin 1"
                ],
            },
        3:{
            'subtask': "Find ‘fork’ and move it to the sinkbasin.", 
            'trajectories':         
                [
                   "go to stove", "open drawer 9", "take fork 1 from drawer 9", "close drawer 9", 
                   "go to sink", "put fork 1 in/on sinkbasin 1"
                ],
            },
        4:{
            'subtask': "Find ‘bowl’ and move it to the sinkbasin.", 
            'trajectories':         
                [
                   "go to microwave", "open cabinet 9", "take bowl 1 from cabinet 9", "close cabinet 9", 
                   "go to sink", "put bowl 1 in/on sinkbasin 1"
                ],
            },
        5:{
            'subtask': "Find ‘spoon’ and move it to the sinkbasin.", 
            'trajectories':         
                [
                   "go to stove", "open drawer 2", "take spoon 1 from drawer 2", "close drawer 2", 
                   "go to sink", "put spoon 1 in/on sinkbasin 1"
                ],
            },
        6:{
            'subtask': "Find ‘ladle’ and move it to the sinkbasin.", 
            'trajectories':         
                [
                   "go to sink", "open drawer 12", "take ladle 1 from drawer 12", "close drawer 12", 
                    "put ladle 1 in/on sinkbasin 1"
                ],
            },
        7:{
            'subtask': "Find ‘dish sponge’ and move it to the sinkbasin.", 
            'trajectories':         
                [
                   "go to stove", "open drawer 7", "take dishsponge 1 from drawer 7", "close drawer 7", 
                   "go to sink", "put dishsponge 1 in/on sinkbasin 1"
                ],
            },
    },
}
