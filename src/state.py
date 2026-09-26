"""GameState — compact immutable-friendly snapshot of a Babylon observation.

Also builds the derived WorldModel used by the tactical brain, opponent model,
MCTS and the runtime policy. All computations are deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geom import PITCH_LENGTH, distance
from .space import SpaceModel
from .wall import WallModel, is_near_wall


@dataclass(frozen=True)
class Player:
    id: str
    team: str  # 'us' | 'them'
    role: str  # 'goalkeeper' | 'outfield'
    x: float
    y: float
    vx: float
    vy: float
    facing: float
    can_act: bool

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @classmethod
    def from_obs(cls, item: dict, team: str) -> "Player":
        return cls(
            id=str(item["id"]),
            team=team,
            role=str(item.get("role", "outfield")),
            x=float(item["position"]["x"]),
            y=float(item["position"]["y"]),
            vx=float(item["velocity"].get("x", 0.0)),
            vy=float(item["velocity"].get("y", 0.0)),
            facing=float(item.get("facingRadians", 0.0)),
            can_act=bool(item.get("canAct", True)),
        )


@dataclass(frozen=True)
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    possessing_team: str | None  # 'us' | 'them' | None
    possessing_player: str | None  # local player id

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @classmethod
    def from_obs(cls, item: dict) -> "Ball":
        pos = item.get("position", {"x": 30.0, "y": 20.0})
        vel = item.get("velocity", {"x": 0.0, "y": 0.0})
        return cls(
            x=float(pos.get("x", 30.0)),
            y=float(pos.get("y", 20.0)),
            vx=float(vel.get("x", 0.0)),
            vy=float(vel.get("y", 0.0)),
            possessing_team=item.get("possessingTeam"),
            possessing_player=item.get("possessedBy"),
        )


@dataclass(frozen=True)
class GameState:
    protocol_version: str
    game_id: str
    sequence: int
    simulation_tick: int
    apply_at_tick: int
    time_remaining: float
    phase: str
    score_us: int
    score_them: int
    ball: Ball
    us: tuple[Player, ...] = field(default_factory=tuple)
    them: tuple[Player, ...] = field(default_factory=tuple)

    @classmethod
    def from_observation(cls, obs: dict) -> "GameState":
        score = obs.get("score", {"us": 0, "them": 0})
        return cls(
            protocol_version=str(obs.get("protocolVersion", "1.0")),
            game_id=str(obs.get("gameId", "")),
            sequence=int(obs.get("sequence", 0)),
            simulation_tick=int(obs.get("simulationTick", 0)),
            apply_at_tick=int(obs.get("applyAtTick", 0)),
            time_remaining=float(obs.get("timeRemainingSeconds", 0.0)),
            phase=str(obs.get("phase", "openPlay")),
            score_us=int(score.get("us", 0)),
            score_them=int(score.get("them", 0)),
            ball=Ball.from_obs(obs.get("ball", {})),
            us=tuple(Player.from_obs(p, "us") for p in obs.get("us", [])),
            them=tuple(Player.from_obs(p, "them") for p in obs.get("them", [])),
        )

    # -- helpers -------------------------------------------------------------
    def outfield_us(self) -> list[Player]:
        return [p for p in self.us if p.role != "goalkeeper"]

    def outfield_them(self) -> list[Player]:
        return [p for p in self.them if p.role != "goalkeeper"]

    def goalkeeper_us(self) -> Player | None:
        return next((p for p in self.us if p.role == "goalkeeper"), None)

    def goalkeeper_them(self) -> Player | None:
        return next((p for p in self.them if p.role == "goalkeeper"), None)

    def our_possessor(self) -> Player | None:
        if self.ball.possessing_team != "us":
            return None
        return next((p for p in self.us if p.id == self.ball.possessing_player), None)

    def their_possessor(self) -> Player | None:
        if self.ball.possessing_team != "them":
            return None
        return next((p for p in self.them if p.id == self.ball.possessing_player), None)

    def nearest_opponent(self, x: float, y: float) -> Player | None:
        if not self.them:
            return None
        return min(self.them, key=lambda p: distance(x, y, p.x, p.y))

    def nearest_teammate(self, player_id: str) -> Player | None:
        src = next((p for p in self.us if p.id == player_id), None)
        if src is None:
            return None
        others = [p for p in self.outfield_us() if p.id != player_id]
        if not others:
            return None
        return min(others, key=lambda p: distance(src.x, src.y, p.x, p.y))

    def has_control(self) -> bool:
        return self.ball.possessing_team == "us"


@dataclass
class WorldModel:
    state: GameState
    ball_zone: str = "mid"  # 'own' | 'mid' | 'final'
    ball_side: str = "center"  # 'left' | 'center' | 'right'
    ball_near_wall: bool = False
    pressure_on_ball: float = 0.0
    our_control: float = 0.0
    their_control: float = 0.0
    goal_x: float = PITCH_LENGTH
    own_goal_x: float = 0.0
    space: SpaceModel = field(default_factory=SpaceModel)
    wall: WallModel = field(default_factory=WallModel)

    @classmethod
    def build(cls, state: GameState, space: SpaceModel | None = None) -> "WorldModel":
        ball = state.ball
        xfrac = ball.x / PITCH_LENGTH
        if xfrac < 0.34:
            zone = "own"
        elif xfrac < 0.66:
            zone = "mid"
        else:
            zone = "final"
        if ball.y < 8.0:
            side = "left"
        elif ball.y > 32.0:
            side = "right"
        else:
            side = "center"
        ours = [p.pos for p in state.outfield_us()]
        theirs = [p.pos for p in state.outfield_them()]
        if space is None:
            space = SpaceModel()
            space.update(ours, theirs)
        else:
            space.update(ours, theirs)
        near = is_near_wall(ball.x, ball.y, margin=4.0)
        opp = state.them
        min_d = min((distance(ball.x, ball.y, p.x, p.y) for p in opp), default=40.0)
        pressure = 0.0
        if min_d < 4.0:
            pressure = (4.0 - min_d) / 4.0
        our_c = space.control_at(ball.x, ball.y)
        return cls(
            state=state,
            ball_zone=zone,
            ball_side=side,
            ball_near_wall=near,
            pressure_on_ball=pressure,
            our_control=our_c,
            their_control=-our_c,
            space=space,
        )

    def ball_speed(self) -> float:
        return math.hypot(self.state.ball.vx, self.state.ball.vy)

    def transitioned_possession(self, previous: "GameState | None") -> bool:
        if previous is None:
            return False
        return previous.has_control() and not self.state.has_control()