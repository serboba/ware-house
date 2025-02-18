import random
import numpy as np
import gym
from gym import spaces
import pygame

from ware_house.classes.Warehouse import Warehouse, Action, Direction, N_GOALS

MAP_STRING = "0,A_2,0,0,0,0,0,0/0,A_3,0,0,0,A_2,0,0/0,0,S,0,0,0,0,0/0,0,0,0,0,0,0,0/0,G,0,0,0,0,0,0"


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, width, height, n_agents, n_shelves):
        self.size = width  # The size of the square grid
        self.window_size = 512
        self.render_mode = "human"
        self.window = None
        self.clock = None

        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.n_shelves = n_shelves
        self.warehouse = Warehouse(height, width)
        self.make_spaces()
        self.reward = 0;

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
            for i in range(self.n_shelves)
        })
        goal_list_space = spaces.Dict({
            i: location_space
            for i in range(N_GOALS)
        })
        self.observation_space = spaces.Dict({
            "agent": agent_list_space,
            "shelf": shelf_list_space,
            "goal": goal_list_space
        })

        self.action_space = spaces.MultiDiscrete([len(Action) for i in range(self.n_agents)])

    def _get_obs(self):
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
        print(type(obs))
        return obs

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
        self.reward = 0
        for i, agent_action in enumerate(action):
            self.warehouse.debug_agents_actions(f'{i + 1}_{agent_action}')
            self.reward += self.warehouse.agent_dict[i + 1].score
        if len(self.warehouse.shelf_dict.keys()) == 0:
            done = True
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, self.reward, done, info

    def render(self, **kwargs):
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
            location = np.array([goal.y, goal.x])
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * location,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agents
        for agent in self.warehouse.agent_dict.values():
            location = np.array([agent.y, agent.x])
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
            if agent.carrying_shelf:
                shelf = self.warehouse.shelf_dict[agent.carrying_shelf_id]
                location = np.array([shelf.y, shelf.x])
                pygame.draw.circle(
                    canvas,
                    (255, 97, 3),
                    (location + 0.5) * pix_square_size,
                    pix_square_size / 4,
                )
        # draw shelves
        for shelf in self.warehouse.free_shelves.values():
            location = np.array([shelf.y, shelf.x])
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (location + 0.5) * pix_square_size,
                pix_square_size / 4,
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

        # spawn shelves
        counter = self.n_shelves
        while counter > 0:
            pos_y = random.randint(0, self.height - 1)
            pos_x = random.randint(0, self.width - 1)
            # shelf spawn format "{pos_y}_{pos_x}_{dir}"
            while not self.warehouse.debug_spawn_shelves(f'{pos_y}_{pos_x}'):
                pos_y = random.randint(0, self.height - 1)
                pos_x = random.randint(0, self.width - 1)
            counter -= 1
        observation = self._get_obs()
        info = self._get_info()
        return observation, info