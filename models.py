from typing import Literal, NotRequired, TypedDict


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


class Observation(TypedDict):
    protocolVersion: str
    gameId: str
    sequence: int
    simulationTick: int
    applyAtTick: int
    timeRemainingSeconds: float
    phase: str
    score: dict[str, int]
    ball: Ball
    us: list[Player]
    them: list[Player]


class Decision(TypedDict):
    protocolVersion: str
    gameId: str
    sequence: int
    intents: list[dict]
