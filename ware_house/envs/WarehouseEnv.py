from os import curdir
import random
import numpy as np
import gym
from gym import spaces
import pygame

from ware_house.classes.Warehouse import Warehouse, Action, Direction, N_GOALS
from utils import SUMMARY_FILE_PATH, REWARDS_FILE_PATH, STEPS_FILE_PATH
MAP_STRING = "0,A_2,0,0,0,0,0,0/0,A_3,0,0,0,A_2,0,0/0,0,S,0,0,0,0,0/0,0,0,0,0,0,0,0/0,G,0,0,0,0,0,0"


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, width, height, n_agents, n_shelves):
        self.size = width  # The size of the square grid
        self.window_size = 512
        self.render_mode = "human"
        self.window = None
        self.clock = None
        self.step_counter = 0
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.cur_shelves_n = n_shelves
        self.N_SHELVES  = n_shelves
        self.warehouse = Warehouse(height, width)
        #self.n_shelves = len(self.warehouse.shelf_dict)
        self.make_spaces()
        self.reward = 0

        self.shelf_previous = {}



    def make_spaces(self):
        location_space = spaces.MultiDiscrete([self.height, self.width])
        # carrying_shelf_space = spaces.MultiBinary(1)
        direction_space = spaces.Discrete(len(Direction))
        # single_agent_attr_spaces = spaces.Dict(
        #   {"location": location_space, "carrying_shelf": carrying_shelf_space, "direction": direction_space})
        single_agent_attr_spaces = spaces.Dict(
            {"location": location_space, "direction": direction_space})
        agent_list_space = spaces.Dict({
            i: single_agent_attr_spaces
            for i in range(self.n_agents)
        })
        shelf_list_space = spaces.Dict({
            i: location_space
            for i in range(self.N_SHELVES)
        })
        goal_list_space = spaces.Dict({
            i: location_space
            for i in range(N_GOALS)
        })
        l1 = []

        for i in range(self.n_agents):
            l1.append(np.array([location_space,direction_space]))
        
        l2= []
        for j in range(self.cur_shelves_n):
            l2.append(location_space)
        l3 = []
        for i in range(N_GOALS):
            l3.append(location_space)


        test1 = np.array([[0,0,0],[10,10,4]])
        test1 = np.tile(test1,len(l1))

        test2 = np.array([[0,0],[10,10]])
        shelftest = np.tile(test2,len(l2))

        #print("shelftest : ", shelftest)
        #print("nshelfs : ", self.N_SHELVES)
        #print(shelftest.shape)
        print("shape? ",shelftest[0].shape)
        goaltest = np.tile(test2,len(l3))

#       test = self._get_obs()
        print("shelffff")
        self.observation_space = spaces.Dict({
            "agent": gym.spaces.Box(low=self.height,high=self.width,shape=(test1[0].shape[0],)),
            "shelf": gym.spaces.Box(low=self.height,high=self.width,shape=(shelftest[0].shape[0],)),
            "goal": gym.spaces.Box(low=self.height,high=self.width,shape=(goaltest[0].shape[0],)),
            "mindist" : gym.spaces.Box(low=0, high=999, shape=(self.n_agents,)),
        })

        '''
        self.observation_space = spaces.Dict({
            "agent": gym.spaces.Box(low=self.height,high=self.width,shape=(test1[0].shape[0],)),
            "shelf": gym.spaces.Box(low=self.height,high=self.width,shape=(shelftest[0].shape[0],)),
            "goal": gym.spaces.Box(low=self.height,high=self.width,shape=(goaltest[0].shape[0],)),
        })
        '''

        self.action_space = spaces.MultiDiscrete([len(Action) for i in range(self.n_agents)])

    def _get_obs(self):
        t1 = []
        t2 = []
        t3 = []
        t4 = []
        for agent in self.warehouse.agent_dict.values():
            t1.append(agent.y)
            t1.append(agent.x)
            t1.append(agent.cur_dir.value)
            t4.append(agent.min_dis)
        
        #dif = self.N_SHELVES - len(self.warehouse.shelf_dict)

        dkeys = list(self.warehouse.shelf_dict.keys())
        mainrange = np.arange(1,self.N_SHELVES+1)
        intersect = np.intersect1d(dkeys,mainrange)

        for i in range (1,mainrange.shape[0]+1):
            if i in intersect:
                t2.append(self.warehouse.shelf_dict[i].y)
                t2.append(self.warehouse.shelf_dict[i].x)
            else:
                t2.append(self.warehouse.goal_dict[1].y)
                t2.append(self.warehouse.goal_dict[1].x)
        
        for goal in self.warehouse.goal_dict.values():
            t3.append (goal.y)
            t3.append (goal.x)
        '''
        obs = spaces.Dict({
            "agent":
                spaces.Dict({
                    i: spaces.Dict({"location": np.array([agent.y, agent.x]),
                                    "direction": agent.cur_dir.value})
                    for i, agent in enumerate(self.warehouse.agent_dict.values())
                }),
            "shelf":
                spaces.Dict({
                    i: spaces.Dict({"location": np.array([shelf.y, shelf.x])})
                    for i, shelf in enumerate(self.warehouse.shelf_dict.values())
                }),
            "goal":
                spaces.Dict({
                    i: spaces.Dict({"location": np.array([goal.y, goal.x])})
                    for i, goal in enumerate(self.warehouse.goal_dict.values())
                })
        })
        '''
        t1 = np.array(t1)
        t2 = np.array(t2)
        t3 = np.array(t3)
        t4 = np.array(t4)

        #print("----t2: ", t2)
        mydict = {
            "agent": t1,
            "shelf" : t2,
            "goal" : t3,
            "mindist" : t4
        }

        return mydict

    def _get_info(self):
        return {
            "agent":
                {
                    i: {"location": np.array([agent.y, agent.x]),
                        "direction": agent.cur_dir.value,
                        "carrying_shelf": agent.carrying_shelf_id,
                        "min dist": agent.min_dis}
                    for i, agent in enumerate(self.warehouse.agent_dict.values())
                },
            "shelf":
                {
                    i: {"location": np.array([shelf.y, shelf.x])}
                    for i, shelf in enumerate(self.warehouse.shelf_dict.values())
                },
            "goal":
                {
                    i: {"location": np.ndarray[goal.y, goal.x]}
                    for i, goal in enumerate(self.warehouse.goal_dict.values())
                }
        }

    def step(self, action):
        done = False
        self.step_counter += 1
        self.reward = 0
        for i, agent_action in enumerate(action):
            self.warehouse.debug_agents_actions(f'{i + 1}_{agent_action}')
            self.reward += self.warehouse.agent_dict[i + 1].score
        if len(self.warehouse.shelf_dict.keys()) == 0:
            done = True
            print(f"Episode ends with {self.step_counter} steps and with reward {self.reward}")
            with open(SUMMARY_FILE_PATH, 'a+') as f:
                f.write(f"(Reward : {self.reward}, Step : {self.step_counter}),")
            with open(REWARDS_FILE_PATH, "a+") as f:
                f.write(f"{self.reward}\n")
            with open(STEPS_FILE_PATH, "a+") as f:
                f.write(f"{self.step_counter}\n")
        else:
            self.cur_shelves_n = len(self.warehouse.shelf_dict.values())
        observation = self._get_obs()
        #info = self._get_info()
        #if self.render_mode == "human":
        #    self._render_frame()
        return observation, self.reward, done, {}

    def render(self, mode, **kwargs):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the goals
        for goal in self.warehouse.goal_dict.values():
            location = np.array([goal.x, goal.y])
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * location,
                    (pix_square_size, pix_square_size),
                ),
            )
        # draw shelves
        for shelf in self.warehouse.free_shelves.values():
            location = np.array([shelf.x, shelf.y])
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (location + 0.5) * pix_square_size,
                pix_square_size / 4,
            )
        # Now we draw the agents
        for agent in self.warehouse.agent_dict.values():
            location = np.array([agent.x, agent.y])
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
            if agent.carrying_shelf:
                shelf = self.warehouse.shelf_dict[agent.carrying_shelf_id]
                location = np.array([shelf.x, shelf.y])
                pygame.draw.circle(
                    canvas,
                    (255, 97, 3),
                    (location + 0.5) * pix_square_size,
                    pix_square_size / 4,
                )

            if agent.cur_dir == Direction.UP:
                agent_dir_start = np.array([agent.x + 0.5, agent.y + 0.25]) * pix_square_size
                agent_dir_end = np.array([agent.x + 0.5, agent.y + 0.50]) * pix_square_size
                pygame.draw.line(
                    canvas,
                    (0, 255, 255),
                    (agent_dir_start[0], agent_dir_start[1]),
                    (agent_dir_end[0], agent_dir_end[1]),
                    width=10,
                )
            elif agent.cur_dir == Direction.RIGHT:
                agent_dir_start = np.array([agent.x + 0.50, agent.y + 0.5]) * pix_square_size
                agent_dir_end = np.array([agent.x + 0.75, agent.y + 0.5]) * pix_square_size
                pygame.draw.line(
                    canvas,
                    (0, 255, 255),
                    (agent_dir_start[0], agent_dir_start[1]),
                    (agent_dir_end[0], agent_dir_end[1]),
                    width=10,
                )
            elif agent.cur_dir == Direction.DOWN:
                agent_dir_start = np.array([agent.x + 0.5, agent.y + 0.5]) * pix_square_size
                agent_dir_end = np.array([agent.x + 0.5, agent.y + 0.75]) * pix_square_size

                pygame.draw.line(
                    canvas,
                    (0, 255, 255),
                    (agent_dir_start[0], agent_dir_start[1]),
                    (agent_dir_end[0], agent_dir_end[1]),
                    width=10,
                )
            else:
                agent_dir_start = np.array([agent.x + 0.25, agent.y + 0.5]) * pix_square_size
                agent_dir_end = np.array([agent.x + 0.5, agent.y + 0.5]) * pix_square_size
                pygame.draw.line(
                    canvas,
                    (0, 255, 255),
                    (agent_dir_start[0], agent_dir_start[1]),
                    (agent_dir_end[0], agent_dir_end[1]),
                    width=10,
                )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def reset(self):
        self.warehouse.reset()
        #print("ooooooooooooooooooooooooooo")
        self.step_counter = 0
        # spawn goals
        counter = N_GOALS
        while counter > 0:
            pos_y = random.randint(0, self.height - 1)
            pos_x = random.randint(0, self.width - 1)
            # goal spawn format "{pos_y}_{pos_x}"
            while not self.warehouse.debug_spawn_goals(f'{pos_y}_{pos_x}'):
                pos_y = random.randint(0, self.height - 1)
                pos_x = random.randint(0, self.width - 1)
            counter -= 1

        # spawn random agents
        counter = self.n_agents
        while counter > 0:
            pos_y = random.randint(0, self.height - 1)
            pos_x = random.randint(0, self.width - 1)
            dir = random.randint(0, len(Direction) - 1)
            # agent spawn format "{pos_y}_{pos_x}_{dir}"
            while not self.warehouse.debug_spawn_agents(f'{pos_y}_{pos_x}_{dir}'):
                pos_y = random.randint(0, self.height - 1)
                pos_x = random.randint(0, self.width - 1)
                dir = random.randint(0, len(Direction) - 1)
            counter -= 1


        # spawn shelves
        counter = self.N_SHELVES
        self.cur_shelves_n = self.N_SHELVES
        while counter > 0:
            pos_y = random.randint(0, self.height - 1)
            pos_x = random.randint(0, self.width - 1)
            # shelf spawn format "{pos_y}_{pos_x}_{dir}"
            while not self.warehouse.debug_spawn_shelves(f'{pos_y}_{pos_x}'):
                pos_y = random.randint(0, self.height - 1)
                pos_x = random.randint(0, self.width - 1)
            counter -= 1
        # print("before!!!!!")
        observation = self._get_obs()
        # print("after!!!!!")
        # info = self._get_info()

        return observation, {}
