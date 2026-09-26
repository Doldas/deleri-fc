from models import Decision, Observation


def decide(observation: Observation) -> Decision:
    """This is the only function a team needs to customize."""
    ball = observation["ball"]["position"]
    outfield = [player for player in observation["us"] if player["role"] != "goalkeeper"]
    chaser = min(
        outfield,
        key=lambda player: (player["position"]["x"] - ball["x"]) ** 2
        + (player["position"]["y"] - ball["y"]) ** 2,
    )["id"]
    lanes = [20.0, 9.0, 31.0, 14.0, 26.0]
    intents = []
    for index, player in enumerate(observation["us"]):
        if player["role"] == "goalkeeper":
            target = {"x": 3.0, "y": max(17.0, min(23.0, ball["y"]))}
        elif player["id"] == chaser:
            target = ball
        else:
            target = {"x": min(49.0, 18.0 + ball["x"] * 0.42), "y": lanes[index]}
        has_ball = observation["ball"].get("possessingTeam") == "us" and observation["ball"].get("possessedBy") == player["id"]
        action = (
            {"type": "none"}
            if not player["canAct"]
            else {"type": "shoot", "target": {"x": 60.0, "y": 20.0}, "power": 0.82}
            if has_ball
            else {"type": "tackle"}
            if observation["ball"].get("possessingTeam") == "them" and player["id"] == chaser
            else {"type": "none"}
        )
        intents.append(
            {
                "playerId": player["id"],
                "move": {"target": target, "speed": 0.85},
                "face": ball,
                "action": action,
            }
        )
    return {
        "protocolVersion": observation["protocolVersion"],
        "gameId": observation["gameId"],
        "sequence": observation["sequence"],
        "intents": intents,
    }
