from typing import Literal, NotRequired, TypedDict

MatchPhase = Literal[
    "kickoff",
    "openPlay",
    "goalCelebration",
    "replay",
    "restart",
    "goldenGoal",
    "penaltyIntro",
    "penaltySetup",
    "penaltyStrike",
    "penaltyResult",
    "completed",
]


class Point(TypedDict):
    x: float
    y: float


class Player(TypedDict):
    id: str
    role: Literal["goalkeeper", "outfield"]
    position: Point
    velocity: Point
    facingRadians: float
    canAct: bool


class Ball(TypedDict):
    id: str
    position: Point
    velocity: Point
    possessedBy: NotRequired[str | None]
    possessingTeam: NotRequired[Literal["us", "them"] | None]


class Score(TypedDict):
    us: int
    them: int


class Observation(TypedDict):
    protocolVersion: str
    gameId: str
    sequence: int
    simulationTick: int
    applyAtTick: int
    timeRemainingSeconds: float
    phase: MatchPhase
    score: Score
    ball: Ball
    us: list[Player]
    them: list[Player]


class Move(TypedDict):
    target: Point
    speed: float


class Action(TypedDict):
    type: Literal["none", "pass", "shoot", "clear", "tackle", "slap"]
    target: NotRequired[Point | None]
    power: NotRequired[float | None]


class Intent(TypedDict):
    playerId: str
    move: NotRequired[Move | None]
    face: NotRequired[Point | None]
    action: NotRequired[Action | None]


class Decision(TypedDict):
    protocolVersion: str
    gameId: str
    sequence: int
    intents: list[Intent]
