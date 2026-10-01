"""TeamPlan — lightweight team-level attacking coordination layer.

This module introduces a shared attacking intention that coordinates off-ball
players, replacing the previous independent per-player decision model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

from .geom import (
    GOAL_CENTER_Y,
    GOAL_HIGH_Y,
    GOAL_LOW_Y,
    OPP_GOAL_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from .state import GameState, Player
from .tactics import (
    ROLE_DEFENDER,
    ROLE_WIDE_LEFT,
    ROLE_WIDE_RIGHT,
    ROLE_STRIKER,
)


class TeamPlanPhase(StrEnum):
    BUILD = "build"              # Ball in own half, building up
    PROGRESSION = "progression"  # Ball in middle third, advancing
    ATTACK = "attack"            # Ball in final third, creating chances
    FINAL_THIRD = "final_third"  # Deep in final third, high-quality opportunities
    SHOT_FOLLOWUP = "shot_followup"  # After our shot, attacking rebounds
    TRANSITION = "transition"    # Lost possession, defensive transition


@dataclass
class TeamPlanTarget:
    """A target position for a specific player role."""
    player_id: str
    role: str                    # 'primary_runner', 'secondary_runner', 'support', 'rest_defender'
    target_x: float
    target_y: float
    priority: float              # Higher = more important to reach
    description: str = ""


@dataclass
class TeamPlan:
    """Shared team attacking intention.
    
    Persists across ticks but has deterministic invalidation conditions.
    """
    phase: TeamPlanPhase = TeamPlanPhase.BUILD
    carrier_id: Optional[str] = None
    created_tick: int = 0
    expires_tick: int = 0
    
    # Role assignments for this plan
    primary_runner_id: Optional[str] = None
    secondary_runner_id: Optional[str] = None
    support_player_id: Optional[str] = None
    rest_defender_id: Optional[str] = None
    
    # Targets for each role
    targets: list[TeamPlanTarget] = field(default_factory=list)
    
    # Shot/rebound context
    shot_expected: bool = False
    shot_committed: bool = False
    rebound_zone_x: Optional[float] = None
    rebound_zone_y: Optional[float] = None
    far_post_target_y: Optional[float] = None
    
    # Plan metadata
    reason: str = ""
    confidence: float = 1.0      # 0-1, how confident we are in this plan
    
    # Invalidation tracking
    last_ball_x: float = 0.0
    last_ball_y: float = 0.0
    last_possession_tick: int = 0


class TeamPlanManager:
    """Manages the team plan lifecycle: create, update, invalidate."""
    
    # Plan timeout in decision ticks (10 Hz) - 3 seconds
    PLAN_TIMEOUT_TICKS = 30
    # Maximum ball movement before plan reconsideration
    BALL_MOVE_THRESHOLD = 8.0
    # Minimum time before allowing plan change
    MIN_PLAN_DURATION = 3  # ticks
    
    def __init__(self):
        self.current_plan: Optional[TeamPlan] = None
        self.previous_plan: Optional[TeamPlan] = None
    
    def update(self, inp, possessor: Optional[Player], current_tick: int) -> Optional[TeamPlan]:
        """Update or create team plan based on current state."""
        state = inp.state
        ball = state.ball
        
        # Check if we should invalidate current plan
        if self.current_plan is not None:
            if self._should_invalidate(inp, possessor, current_tick, ball):
                self.previous_plan = self.current_plan
                self.current_plan = None
        
        # Create new plan if needed
        if self.current_plan is None and state.has_control() and possessor is not None:
            self.current_plan = self._create_plan(inp, possessor, current_tick, ball)
        elif self.current_plan is not None:
            # Refine existing plan
            self._refine_plan(inp, possessor, current_tick, ball)
        
        return self.current_plan
    
    def _should_invalidate(self, inp, possessor: Optional[Player], current_tick: int, ball) -> bool:
        """Deterministic invalidation conditions."""
        plan = self.current_plan
        
        # 1. Possession lost
        if not inp.state.has_control():
            return True
        
        # 2. Carrier changed
        if possessor is not None and self.current_plan is not None and self.current_plan.carrier_id != possessor.id:
            return True
        
        # 3. Ball trajectory changed materially (pass/shot played)
        if self.current_plan is not None:
            ball_move = math.hypot(ball.x - self.current_plan.last_ball_x, ball.y - self.current_plan.last_ball_y)
            if ball_move > self.BALL_MOVE_THRESHOLD:
                # Check if it's a pass (ball speed high) vs carrier dribbling
                ball_speed = math.hypot(ball.vx, ball.vy)
                if ball_speed > 10.0:  # Pass or shot
                    return True
        
        # 4. Intended receiver no longer viable
        if self.current_plan is not None and self.current_plan.primary_runner_id:
            runner = next((p for p in inp.state.outfield_us() if p.id == self.current_plan.primary_runner_id), None)
            if runner is None or not runner.can_act:
                return True
            # Receiver too far from expected position or marked
            if runner.x < ball.x - 5.0:  # Behind ball
                return True
        
        # 5. Loose ball began
        if self.current_plan is not None and ball.possessing_team is None and self.current_plan.last_possession_tick > 0:
            return True
        
        # 6. Shot occurred
        if self.current_plan is not None and self.current_plan.shot_committed and ball.possessing_team is None:
            # Transition to shot followup handled separately
            return True
        
        # 7. Plan timeout
        if self.current_plan is not None and current_tick - self.current_plan.created_tick > self.PLAN_TIMEOUT_TICKS:
            return True
        
        # 8. Dangerous transition requires defensive priority
        if self.current_plan is not None and self._high_transition_risk(inp):
            return True
        
        # 9. Game phase changed significantly
        if self.current_plan is not None and self._phase_changed(inp, self.current_plan):
            return True
        
        return False
    
    def _high_transition_risk(self, inp) -> bool:
        """Check if counterattack risk is high."""
        state = inp.state
        ball = state.ball
        if ball.x > 40.0:  # We're deep in their half
            # Count our players ahead of ball
            ahead = sum(1 for p in state.outfield_us() if p.x > ball.x)
            # Count their players behind ball
            behind = sum(1 for p in state.outfield_them() if p.x < ball.x)
            if ahead >= 3 and behind <= 1:
                return True  # Overcommitted
        return False
    
    def _phase_changed(self, inp, plan: TeamPlan) -> bool:
        """Check if tactical phase changed enough to warrant new plan."""
        state = inp.state
        ball = state.ball
        
        new_phase = self._determine_phase(state, ball)
        if new_phase != plan.phase:
            # Allow phase progression but not regression within attack
            phase_order = [
                TeamPlanPhase.BUILD,
                TeamPlanPhase.PROGRESSION,
                TeamPlanPhase.ATTACK,
                TeamPlanPhase.FINAL_THIRD,
            ]
            try:
                old_idx = phase_order.index(plan.phase)
                new_idx = phase_order.index(new_phase)
                if new_idx < old_idx:  # Regression
                    return True
            except ValueError:
                return True
        return False
    
    def _determine_phase(self, state: GameState, ball) -> TeamPlanPhase:
        """Determine attacking phase from ball position and context."""
        if ball.x < 20.0:
            return TeamPlanPhase.BUILD
        elif ball.x < 35.0:
            return TeamPlanPhase.PROGRESSION
        elif ball.x < 45.0:
            return TeamPlanPhase.ATTACK
        else:
            return TeamPlanPhase.FINAL_THIRD
    
    def _create_plan(self, inp, possessor: Player, current_tick: int, ball) -> TeamPlan:
        """Create a new team plan from scratch."""
        state = inp.state
        roles = inp.roles
        world = inp.world
        
        # Determine phase
        phase = self._determine_phase(state, ball)
        
        # Select complementary attacking roles based on geometry
        primary, secondary, support, rest = self._assign_attacking_roles(
            inp, possessor, phase
        )
        
        # Generate targets for each role
        targets = self._generate_targets(inp, possessor, primary, secondary, support, rest, phase)
        
        # Build reason string for debugging
        reason = self._build_reason(inp, possessor, primary, secondary, phase)
        
        plan = TeamPlan(
            phase=phase,
            carrier_id=possessor.id,
            created_tick=current_tick,
            expires_tick=current_tick + self.PLAN_TIMEOUT_TICKS,
            primary_runner_id=primary.id if primary else None,
            secondary_runner_id=secondary.id if secondary else None,
            support_player_id=support.id if support else None,
            rest_defender_id=rest.id if rest else None,
            targets=targets,
            reason=reason,
            confidence=0.8,
            last_ball_x=ball.x,
            last_ball_y=ball.y,
            last_possession_tick=current_tick,
        )
        
        return plan
    
    def _assign_attacking_roles(
        self, inp, possessor: Player, phase: TeamPlanPhase
    ) -> tuple[Optional[Player], Optional[Player], Optional[Player], Optional[Player]]:
        """Assign complementary attacking roles based on geometry, not formation.
        
        Returns: (primary_runner, secondary_runner, support, rest_defender)
        """
        state = inp.state
        our_players = [p for p in state.outfield_us() if p.id != possessor.id]
        
        if not our_players:
            return None, None, None, None
        
        # Score each player for each role
        primary_scores = []
        secondary_scores = []
        support_scores = []
        rest_scores = []
        
        # Determine max attacking roles based on available players
        # Always reserve at least 1 for rest defense
        max_attacking_roles = max(1, len(our_players) - 1)
        
        # First pass: score all players for each role
        primary_candidates = []
        for p in our_players:
            primary_score = self._score_primary_runner(inp, possessor, p, phase)
            primary_candidates.append((primary_score, p))
        
        primary_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Primary runner gets first pick
        assigned = set()
        primary = secondary = support = rest = None
        
        for score, p in primary_candidates:
            if score > 0.1 and p.id not in assigned:
                primary = p
                assigned.add(p.id)
                break
        
        # Secondary runner: score with knowledge of primary
        if len(assigned) < max_attacking_roles:
            secondary_candidates = []
            for p in our_players:
                if p.id in assigned:
                    continue
                secondary_score = self._score_secondary_runner(inp, possessor, p, phase, primary)
                secondary_candidates.append((secondary_score, p))
            
            secondary_candidates.sort(key=lambda x: x[0], reverse=True)
            
            for score, p in secondary_candidates:
                if score > 0.1 and p.id not in assigned:
                    secondary = p
                    assigned.add(p.id)
                    break
        
        # Support - only if we have enough players
        if len(assigned) < max_attacking_roles:
            support_candidates = []
            for p in our_players:
                if p.id in assigned:
                    continue
                support_score = self._score_support(inp, possessor, p, phase)
                support_candidates.append((support_score, p))
            
            support_candidates.sort(key=lambda x: x[0], reverse=True)
            
            for score, p in support_candidates:
                if score > 0.1 and p.id not in assigned:
                    support = p
                    assigned.add(p.id)
                    break
        
        # Rest defender (must have one)
        rest_candidates = []
        for p in our_players:
            if p.id in assigned:
                continue
            rest_score = self._score_rest_defender(inp, possessor, p, phase)
            rest_candidates.append((rest_score, p))
        
        rest_candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, p in rest_candidates:
            if p.id not in assigned:
                rest = p
                assigned.add(p.id)
                break
        
        # If no rest defender assigned, pick the deepest player
        if rest is None and our_players:
            remaining = [p for p in our_players if p.id not in assigned]
            if remaining:
                rest = min(remaining, key=lambda p: p.x)  # Most defensive
        
        return primary, secondary, support, rest
    
    def _score_primary_runner(self, inp, possessor: Player, p: Player, phase: TeamPlanPhase) -> float:
        """Score player for primary runner role (depth threat)."""
        state = inp.state
        ball = state.ball
        
        score = 0.0
        
        # Must be able to act
        if not p.can_act:
            return -1.0
        
        # Ahead of carrier is good for depth runs
        if p.x > possessor.x:
            score += (p.x - possessor.x) * 0.1
        else:
            # Behind carrier can still run forward if space
            score -= (possessor.x - p.x) * 0.05
        
        # Space behind defensive line
        their_deepest = min((q.x for q in state.outfield_them()), default=PITCH_LENGTH)
        space_behind = their_deepest - p.x
        if space_behind > 5.0:
            score += space_behind * 0.2
        
        # In final third, prefer players already high
        if phase in (TeamPlanPhase.ATTACK, TeamPlanPhase.FINAL_THIRD):
            if p.x > 40.0:
                score += 10.0
        
        # Wide players good for diagonal runs
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
            score += 5.0
        
        # Striker natural fit
        if role == ROLE_STRIKER:
            score += 8.0
        
        # Penalize if tightly marked
        nearest_opp = min(
            (math.hypot(p.x - o.x, p.y - o.y) for o in state.outfield_them()),
            default=20.0
        )
        if nearest_opp < 3.0:
            score -= 10.0
        
        return score
    
    def _score_secondary_runner(self, inp, possessor: Player, p: Player, phase: TeamPlanPhase, primary: Optional[Player] = None) -> float:
        """Score player for secondary runner (width/far-post/cutback)."""
        state = inp.state
        
        if not p.can_act:
            return -1.0
        
        score = 0.0
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        
        # Wide players natural for width
        if role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
            score += 10.0
            # On correct flank for width
            if role == ROLE_WIDE_LEFT and p.y < GOAL_CENTER_Y:
                score += 5.0
            elif role == ROLE_WIDE_RIGHT and p.y > GOAL_CENTER_Y:
                score += 5.0
        
        # In final third, far-post runs valuable
        if phase in (TeamPlanPhase.ATTACK, TeamPlanPhase.FINAL_THIRD):
            # Far post opposite to carrier side
            far_post_y = GOAL_LOW_Y if possessor.y > GOAL_CENTER_Y else GOAL_HIGH_Y
            if abs(p.y - far_post_y) < 10.0:
                score += 8.0
        
        # Not same space as primary
        if primary is not None:
            if math.hypot(p.x - primary.x, p.y - primary.y) < 5.0:
                score -= 5.0
        
        return score
    
    def _score_support(self, inp, possessor: Player, p: Player, phase: TeamPlanPhase) -> float:
        """Score player for support role (triangle, cutback, recycle)."""
        state = inp.state
        
        if not p.can_act:
            return -1.0
        
        score = 0.0
        dist_to_carrier = math.hypot(p.x - possessor.x, p.y - possessor.y)
        
        # Close to carrier for short options
        if dist_to_carrier < 10.0:
            score += (10.0 - dist_to_carrier) * 0.5
        
        # Behind or level with carrier for support
        if p.x <= possessor.x + 3.0:
            score += 5.0
        
        # Central position for cutbacks
        if abs(p.y - GOAL_CENTER_Y) < 8.0:
            score += 3.0
        
        # Midfielders/defenders natural support
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        if role == ROLE_DEFENDER:
            score += 3.0
        
        return score
    
    def _score_rest_defender(self, inp, possessor: Player, p: Player, phase: TeamPlanPhase) -> float:
        """Score player for rest defender (transition protection)."""
        state = inp.state
        
        if not p.can_act:
            return -1.0
        
        score = 0.0
        
        # Behind ball is mandatory
        if p.x < possessor.x:
            score += (possessor.x - p.x) * 0.2
        else:
            score -= 20.0  # Strongly penalize being ahead
        
        # Central for coverage
        if abs(p.y - GOAL_CENTER_Y) < 5.0:
            score += 5.0
        
        # Defenders natural
        role = inp.roles.get(p.id, ROLE_DEFENDER)
        if role == ROLE_DEFENDER:
            score += 10.0
        
        # In final third, must have rest defender
        if phase in (TeamPlanPhase.ATTACK, TeamPlanPhase.FINAL_THIRD):
            score += 10.0
        
        return score
    
    def _generate_targets(
        self, inp, possessor: Optional[Player],
        primary, secondary, support, rest,
        phase: TeamPlanPhase
    ) -> list[TeamPlanTarget]:
        """Generate target positions for each assigned role."""
        targets = []
        state = inp.state
        ball = state.ball
        # Opponent's defensive line (closest to their goal, max x)
        opp_defensive_line = max((q.x for q in state.outfield_them()), default=0.0)
        # Opponent's deepest attacker (closest to our goal, min x)
        opp_deepest_attacker = min((q.x for q in state.outfield_them()), default=PITCH_LENGTH)
        
        # Primary runner target: depth behind defensive line
        if primary:
            if phase in (TeamPlanPhase.ATTACK, TeamPlanPhase.FINAL_THIRD):
                # Run in behind opponent's defensive line
                tx = min(opp_defensive_line + 5.0, OPP_GOAL_X - 2.0)
                # Diagonal run toward far post
                carrier_y = possessor.y if possessor else GOAL_CENTER_Y
                far_post_y = GOAL_LOW_Y if carrier_y > GOAL_CENTER_Y else GOAL_HIGH_Y
                ty = far_post_y
            elif phase == TeamPlanPhase.PROGRESSION:
                # Push defensive line
                tx = min(opp_defensive_line + 2.0, 45.0)
                ty = self._pick_channel_y(inp, primary, tx)
            else:
                # Build up: stay high as outlet
                carrier_x = possessor.x if possessor else 30.0
                tx = max(carrier_x + 10.0, 35.0)
                ty = self._pick_channel_y(inp, primary, tx)
            
            targets.append(TeamPlanTarget(
                player_id=primary.id,
                role="primary_runner",
                target_x=tx,
                target_y=ty,
                priority=1.0,
                description=f"depth_run_to_{tx:.0f},{ty:.0f}"
            ))
        
        # Secondary runner target: width / far post / cutback
        if secondary:
            role = inp.roles.get(secondary.id, ROLE_DEFENDER)
            if phase in (TeamPlanPhase.ATTACK, TeamPlanPhase.FINAL_THIRD) and role in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
                # Far post run
                tx = min(opp_defensive_line + 3.0, OPP_GOAL_X - 3.0)
                ty = GOAL_LOW_Y if role == ROLE_WIDE_LEFT else GOAL_HIGH_Y
                desc = "far_post_run"
            elif phase == TeamPlanPhase.FINAL_THIRD and role == ROLE_STRIKER:
                # Cutback zone
                tx = max(ball.x - 5.0, 40.0)
                ty = GOAL_CENTER_Y
                desc = "cutback_zone"
            else:
                # Width maintenance
                tx = max(ball.x, 25.0)
                ty = 5.0 if role == ROLE_WIDE_LEFT else 35.0
                if role not in (ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT):
                    carrier_y = possessor.y if possessor else GOAL_CENTER_Y
                    ty = GOAL_CENTER_Y + (5.0 if carrier_y < GOAL_CENTER_Y else -5.0)
                desc = "width_hold"
            
            targets.append(TeamPlanTarget(
                player_id=secondary.id,
                role="secondary_runner",
                target_x=tx,
                target_y=ty,
                priority=0.8,
                description=desc
            ))
        
        # Support target: triangle / cutback zone
        if support:
            if phase in (TeamPlanPhase.ATTACK, TeamPlanPhase.FINAL_THIRD):
                # Cutback zone at edge of box
                tx = max(ball.x - 8.0, 38.0)
                ty = GOAL_CENTER_Y
                desc = "cutback_support"
            else:
                # Triangle support
                carrier_x = possessor.x if possessor else 30.0
                carrier_y = possessor.y if possessor else GOAL_CENTER_Y
                tx = carrier_x - 3.0
                ty = carrier_y + (5.0 if carrier_y < GOAL_CENTER_Y else -5.0)
                desc = "triangle_support"
            
            targets.append(TeamPlanTarget(
                player_id=support.id,
                role="support",
                target_x=tx,
                target_y=ty,
                priority=0.6,
                description=desc
            ))
        
        # Rest defender target: transition cover
        if rest:
            # Goal-side of ball, central
            carrier_x = possessor.x if possessor else ball.x
            tx = min(carrier_x * 0.5, 25.0)
            tx = max(tx, 8.0)
            ty = GOAL_CENTER_Y
            
            # Shade to ball side
            if ball.y < GOAL_CENTER_Y:
                ty -= 5.0
            elif ball.y > GOAL_CENTER_Y:
                ty += 5.0
            
            targets.append(TeamPlanTarget(
                player_id=rest.id,
                role="rest_defender",
                target_x=tx,
                target_y=ty,
                priority=0.9,  # High priority for safety
                description="transition_cover"
            ))
        
        return targets
    
    def _pick_channel_y(self, inp, player: Player, target_x: float) -> float:
        """Pick Y channel for striker run between CB and FB."""
        state = inp.state
        
        # Find gaps in their defensive line
        left_gap = 12.0   # Between left CB and left FB
        right_gap = 28.0  # Between right CB and right FB
        
        # Count defenders in each channel
        left_defenders = sum(1 for q in state.outfield_them() 
                           if q.x > target_x - 5.0 and q.y < GOAL_CENTER_Y)
        right_defenders = sum(1 for q in state.outfield_them() 
                            if q.x > target_x - 5.0 and q.y > GOAL_CENTER_Y)
        
        # Prefer less defended channel
        if left_defenders < right_defenders:
            return left_gap
        elif right_defenders < left_defenders:
            return right_gap
        else:
            # Default to player's current side
            return left_gap if player.y < GOAL_CENTER_Y else right_gap
    
    def _build_reason(self, inp, possessor: Player, primary, secondary, phase: TeamPlanPhase) -> str:
        """Build human-readable reason for this plan."""
        parts = [f"phase={phase.value}", f"carrier={possessor.id}"]
        if primary:
            parts.append(f"primary={primary.id}")
        if secondary:
            parts.append(f"secondary={secondary.id}")
        # Add tactical context
        state = inp.state
        their_deepest = min((q.x for q in state.outfield_them()), default=PITCH_LENGTH)
        if their_deepest > 40.0:
            parts.append("high_line")
        elif their_deepest < 25.0:
            parts.append("low_block")
        return " ".join(parts)
    
    def _refine_plan(self, inp, possessor: Optional[Player], current_tick: int, ball) -> None:
        """Refine existing plan based on new state."""
        if self.current_plan is None:
            return
        
        plan = self.current_plan
        state = inp.state
        
        # Update ball position tracking
        plan.last_ball_x = ball.x
        plan.last_ball_y = ball.y
        plan.last_possession_tick = current_tick
        
        # Update phase if progressed
        new_phase = self._determine_phase(state, ball)
        if new_phase != plan.phase:
            plan.phase = new_phase
            # Regenerate targets for new phase
            primary = next((p for p in state.outfield_us() if p.id == plan.primary_runner_id), None)
            secondary = next((p for p in state.outfield_us() if p.id == plan.secondary_runner_id), None)
            support = next((p for p in state.outfield_us() if p.id == plan.support_player_id), None)
            rest = next((p for p in state.outfield_us() if p.id == plan.rest_defender_id), None)
            plan.targets = self._generate_targets(inp, possessor, primary, secondary, support, rest, new_phase)
        
        # Handle shot commitment
        # Check if carrier is about to shoot (high EV shot available)
        if possessor and possessor.can_act:
            from .policy import ExpectedValueCalculator, _shot_geometry_target, _shot_worth_taking
            if _shot_worth_taking(inp, possessor):
                (tx, ty), power = _shot_geometry_target(inp, possessor)
                ev = ExpectedValueCalculator(inp).ev_shot(possessor, (tx, ty), power)
                if ev > 10.0:  # High-value shot
                    plan.shot_expected = True
                    plan.rebound_zone_x = tx
                    plan.rebound_zone_y = ty
                    # Far post for rebound
                    plan.far_post_target_y = GOAL_LOW_Y if ty > GOAL_CENTER_Y else GOAL_HIGH_Y


# Global instance for per-match lifecycle
_team_plan_managers: dict[str, TeamPlanManager] = {}


def get_team_plan_manager(game_id: str) -> TeamPlanManager:
    """Get or create TeamPlanManager for a match."""
    if game_id not in _team_plan_managers:
        _team_plan_managers[game_id] = TeamPlanManager()
    return _team_plan_managers[game_id]


def clear_team_plan_manager(game_id: str) -> None:
    """Clear TeamPlanManager at match end."""
    _team_plan_managers.pop(game_id, None)


def reset_team_plan_manager(game_id: str) -> None:
    """Reset the current plan in TeamPlanManager (for test mode)."""
    if game_id in _team_plan_managers:
        _team_plan_managers[game_id].current_plan = None
        _team_plan_managers[game_id].previous_plan = None