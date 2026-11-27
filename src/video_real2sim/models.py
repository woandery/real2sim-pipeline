"""Typed contracts shared by model calls, the controller and the USD compiler.

Models may propose data only. No generated Python, shell, paths or URLs are accepted.
Cross-reference checks run locally even when the API enforces the JSON schema.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,95}$")]
Vector = Annotated[list[float], Field(min_length=3, max_length=3)]
Frames = Annotated[list[int], Field(min_length=1)]
Color = Annotated[list[Annotated[float, Field(ge=0, le=1)]], Field(min_length=3, max_length=3)]
Positive = Annotated[float, Field(gt=0, lt=20)]
Size = Annotated[list[Positive], Field(min_length=3, max_length=3)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Environment(Contract):
    category: Literal["tabletop", "vertical_workspace", "floor_assembly", "free_space"]
    rationale: str
    support_surface: str
    camera_motion: str


class ObservedObject(Contract):
    id: Identifier
    name_zh: str
    role: Literal["target", "container", "support", "obstacle", "visual_context"]
    visible_at_start: bool
    first_visible_frame: int = Field(ge=0)
    evidence_frames: Frames
    initial_state: str
    final_state: str
    geometry_observations: str
    material_observations: str
    confidence: float = Field(ge=0, le=1)
    uncertainty: str


class Action(Contract):
    name_zh: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    participants: list[Identifier]
    observed_precondition: str
    observed_postcondition: str
    evidence_frames: Frames
    confidence: float = Field(ge=0, le=1)
    uncertainty: str


class HiddenObject(Contract):
    object_id: Identifier
    revealed_frames: Frames
    inferred_initial_location: str
    basis: str
    ambiguity: str


class EvidenceRequest(Contract):
    interval_frames: list[int] = Field(min_length=2, max_length=2)
    reason: str
    requested_view: str


class Understanding(Contract):
    scene_id: Identifier
    summary_zh: str
    environment: Environment
    objects: list[ObservedObject] = Field(min_length=1)
    actions: list[Action] = Field(min_length=1)
    hidden_object_reasoning: list[HiddenObject]
    initial_scene_state: str
    final_scene_state: str
    task_success_predicates: list[str] = Field(min_length=1)
    missing_evidence: list[EvidenceRequest]
    reconstruction_priorities: list[Identifier]

    def validate_video(
        self, scene_id: str, frame_count: int, observed_frames: set[int] | None = None
    ) -> None:
        if self.scene_id != scene_id:
            raise ValueError("Understanding refers to a different scene")
        names = [obj.id for obj in self.objects]
        if len(names) != len(set(names)):
            raise ValueError("Object identities must be unique")
        references = [*self.reconstruction_priorities]
        frames = []
        for obj in self.objects:
            frames.extend([obj.first_visible_frame, *obj.evidence_frames])
        for action in self.actions:
            if action.end_frame < action.start_frame:
                raise ValueError("Action interval is reversed")
            references.extend(action.participants)
            frames.extend([action.start_frame, action.end_frame, *action.evidence_frames])
        for obj in self.hidden_object_reasoning:
            references.append(obj.object_id)
            frames.extend(obj.revealed_frames)
        for request in self.missing_evidence:
            if request.interval_frames[0] > request.interval_frames[1]:
                raise ValueError("Requested interval is reversed")
            frames.extend(request.interval_frames)
        if set(references) - set(names):
            raise ValueError("Understanding references unknown objects")
        if any(not 0 <= frame < frame_count for frame in frames):
            raise ValueError("Evidence frame is outside the video")
        cited = {f for obj in self.objects for f in [obj.first_visible_frame, *obj.evidence_frames]}
        cited.update(f for action in self.actions for f in action.evidence_frames)
        cited.update(f for obj in self.hidden_object_reasoning for f in obj.revealed_frames)
        if observed_frames is not None and cited - observed_frames:
            raise ValueError("Understanding cites source frames that were not supplied")


class Camera(Contract):
    position: Vector
    target: Vector
    vertical_fov_deg: float = Field(gt=1, lt=170)


class Orbit(Contract):
    target: Vector
    radius: Positive
    height: Positive


class Ground(Contract):
    size: Size
    position: Vector
    color: Color


class HullSettings(Contract):
    hull_vertex_limit: int = Field(ge=4, le=255)
    min_thickness: float = Field(ge=0, le=0.001)


class Part(Contract):
    id: Identifier
    shape: Literal["box", "sphere", "cylinder", "capsule", "cone", "lathe"]
    size: Size
    position: Vector
    rotation_deg: Vector
    color: Color
    collision: bool
    opacity: float = Field(ge=0, le=1)
    profile: list[Annotated[list[float], Field(min_length=2, max_length=2)]] | None = None
    segments: int | None = Field(default=None, ge=12, le=128)
    convex_sectors: int | None = Field(default=None, ge=4, le=96)
    convex_hull: HullSettings | None = None
    contact_offset_m: float | None = Field(default=None, gt=0, le=0.02)
    rest_offset_m: float | None = Field(default=None, gt=-0.01)

    @model_validator(mode="after")
    def valid_geometry(self) -> Part:
        if self.shape == "lathe":
            if not self.profile or len(self.profile) < 2:
                raise ValueError("Lathe requires a radial profile")
            if any(not 0 <= point[0] < 20 for point in self.profile):
                raise ValueError("Invalid profile radius")
        elif (
            self.profile is not None or self.segments is not None or self.convex_sectors is not None
        ):
            raise ValueError("Profile/sector settings require a lathe")
        if self.convex_sectors is not None:
            if not self.collision or self.opacity != 0:
                raise ValueError("Convex sectors must be invisible collision geometry")
            if len(self.profile or []) < 4 or self.profile[0] != self.profile[-1]:
                raise ValueError("Convex sector profile must be closed")
        if self.convex_hull is not None and (not self.collision or self.shape != "lathe"):
            raise ValueError("Hull settings require a colliding lathe")
        if self.contact_offset_m is not None or self.rest_offset_m is not None:
            if not self.collision or self.contact_offset_m is None or self.rest_offset_m is None:
                raise ValueError("Both offsets are required for collision geometry")
            if self.rest_offset_m >= self.contact_offset_m:
                raise ValueError("Rest offset must be smaller than contact offset")
        return self


class Asset(Contract):
    id: Identifier
    label_zh: str
    source_object_id: Identifier
    evidence_frames: Frames
    body_type: Literal["static", "dynamic", "kinematic"]
    mass_kg: float = Field(gt=0, le=10000)
    position: Vector
    rotation_deg: Vector
    parts: list[Part] = Field(min_length=1, max_length=256)
    reconstruction_note: str


class Drive(Contract):
    stiffness: float = Field(ge=0)
    damping: float = Field(ge=0)
    max_force: float = Field(ge=0)


class Joint(Contract):
    id: Identifier
    type: Literal["prismatic", "revolute", "fixed"]
    parent: Identifier
    child: Identifier
    axis: Literal["X", "Y", "Z"]
    lower: float
    upper: float
    parent_anchor: Vector
    child_anchor: Vector
    initial_position: float
    drive: Drive | None
    collision_enabled: bool


class Pose(Contract):
    asset_id: Identifier
    position: Vector
    rotation_deg: Vector


class JointPosition(Contract):
    joint_id: Identifier
    value: float


class DiagnosticState(Contract):
    name_zh: str
    kind: Literal["initial", "reveal", "final"]
    evidence_frames: Frames
    poses: list[Pose]
    joint_positions: list[JointPosition]
    enabled_attachments: list[Identifier] = Field(default_factory=list)
    note: str


class Attachment(Contract):
    id: Identifier
    parent: Identifier
    child: Identifier
    parent_anchor: Vector
    child_anchor: Vector
    collision_enabled: bool
    break_force_N: float = Field(gt=0)
    break_torque_Nm: float = Field(gt=0)
    reference_state_index: int = Field(ge=0)
    approximation_note: str = Field(min_length=1)


def unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")


class SceneSpec(Contract):
    scene_id: Identifier
    compiler_version: Literal[5] = 5
    units: Literal["meters"]
    up_axis: Literal["Z"]
    scale_assumptions: list[str] = Field(min_length=1)
    geometry_limitations: list[str]
    camera: Camera
    orbit: Orbit
    ground: Ground
    assets: list[Asset] = Field(min_length=1, max_length=128)
    joints: list[Joint]
    attachments: list[Attachment] = Field(default_factory=list)
    diagnostic_states: list[DiagnosticState] = Field(min_length=2, max_length=64)
    action_summary_zh: str
    final_state_description: str

    @model_validator(mode="after")
    def references_and_constraints(self) -> SceneSpec:
        unique([asset.id for asset in self.assets], "asset ID")
        unique([j.id for j in self.joints], "joint ID")
        unique([a.id for a in self.attachments], "attachment ID")
        assets = {asset.id: asset for asset in self.assets}
        joints = {j.id: j for j in self.joints}
        attachments = {a.id: a for a in self.attachments}
        for asset in self.assets:
            unique([part.id for part in asset.parts], "part ID")
            if asset.body_type == "dynamic" and not any(p.collision for p in asset.parts):
                raise ValueError("Dynamic assets need collision geometry")
        children = []
        for joint in self.joints:
            if (
                joint.parent not in assets
                or joint.child not in assets
                or joint.parent == joint.child
            ):
                raise ValueError("Invalid joint body reference")
            if assets[joint.child].body_type != "dynamic":
                raise ValueError("An articulated child must remain dynamic")
            if not joint.lower <= joint.initial_position <= joint.upper:
                raise ValueError("Initial joint value is outside limits")
            # Compiler applies states in joint order; reject cycles and out-of-order chains.
            children.append(joint.child)
        unique(children, "joint child")
        for i, joint in enumerate(self.joints):
            if joint.parent in children[i:]:
                raise ValueError("Joints must form a parent-first acyclic forest")
        if (
            self.diagnostic_states[0].kind != "initial"
            or self.diagnostic_states[-1].kind != "final"
        ):
            raise ValueError("States must begin with initial and end with final")
        if sum(s.kind == "initial" for s in self.diagnostic_states) != 1:
            raise ValueError("Exactly one initial state is required")
        for state in self.diagnostic_states:
            unique([p.asset_id for p in state.poses], "state pose")
            unique([p.joint_id for p in state.joint_positions], "state joint")
            if any(p.asset_id not in assets for p in state.poses):
                raise ValueError("Unknown state asset")
            for pos in state.joint_positions:
                joint = joints.get(pos.joint_id)
                if joint is None or not joint.lower <= pos.value <= joint.upper:
                    raise ValueError("Unknown or out-of-range state joint")
            if set(state.enabled_attachments) - attachments.keys():
                raise ValueError("Unknown enabled attachment")
        initial = self.diagnostic_states[0]
        if initial.poses or initial.joint_positions:
            raise ValueError("Initial state must use the asset poses and joint initial values")
        for attachment in self.attachments:
            if attachment.parent not in assets or attachment.child not in assets:
                raise ValueError("Unknown attachment body")
            if attachment.parent == attachment.child:
                raise ValueError("Self attachment is invalid")
            if any(assets[n].body_type != "dynamic" for n in (attachment.parent, attachment.child)):
                raise ValueError("Assembly parts must be dynamic")
            if attachment.reference_state_index >= len(self.diagnostic_states):
                raise ValueError("Attachment reference state is out of range")
            reference = self.diagnostic_states[attachment.reference_state_index]
            if attachment.id not in reference.enabled_attachments:
                raise ValueError("Attachment must be enabled in its reference state")
        self.validate_joint_geometry()
        return self

    def validate_joint_geometry(self) -> None:
        """Check actual local anchors before allowing the GPU worker to start."""
        import numpy as np
        from scipy.spatial.transform import Rotation

        assets = {asset.id: asset for asset in self.assets}
        for joint in self.joints:
            parent, child = assets[joint.parent], assets[joint.child]
            rp = Rotation.from_euler("xyz", parent.rotation_deg, degrees=True)
            rc = Rotation.from_euler("xyz", child.rotation_deg, degrees=True)
            p0 = np.array(parent.position) + rp.apply(joint.parent_anchor)
            p1 = np.array(child.position) + rc.apply(joint.child_anchor)
            if joint.type == "prismatic":
                p0 += rp.apply(np.eye(3)["XYZ".index(joint.axis)]) * joint.initial_position
            if np.linalg.norm(p1 - p0) > 0.0005:
                raise ValueError(f"Joint anchors do not coincide: {joint.id}")

        def relation(attachment, state):
            poses = {asset.id: asset for asset in self.assets}
            poses.update({pose.asset_id: pose for pose in state.poses})
            parent, child = poses[attachment.parent], poses[attachment.child]
            rp = Rotation.from_euler("xyz", parent.rotation_deg, degrees=True)
            rc = Rotation.from_euler("xyz", child.rotation_deg, degrees=True)
            delta = (
                np.array(parent.position)
                + rp.apply(attachment.parent_anchor)
                - np.array(child.position)
                - rc.apply(attachment.child_anchor)
            )
            return float(np.linalg.norm(delta)), rp.inv() * rc

        for attachment in self.attachments:
            reference = self.diagnostic_states[attachment.reference_state_index]
            _, relative = relation(attachment, reference)
            for state in self.diagnostic_states:
                if attachment.id in state.enabled_attachments:
                    error, rotation = relation(attachment, state)
                    if error > 0.0005 or (relative.inv() * rotation).magnitude() > np.deg2rad(0.5):
                        raise ValueError(f"Enabled attachment is inconsistent: {attachment.id}")

    def compiler_dict(self) -> dict:
        result = self.model_dump(exclude_none=True)
        # None is semantically significant: it disables the historical default drive.
        for target, source in zip(result["joints"], self.joints, strict=True):
            target["drive"] = source.drive.model_dump() if source.drive is not None else None
        return result

    def validate_evidence(
        self,
        understanding: Understanding,
        frame_count: int,
        observed_frames: set[int] | None = None,
    ) -> None:
        if self.scene_id != understanding.scene_id:
            raise ValueError("Scene identity changed")
        source_ids = {obj.id for obj in understanding.objects}
        if any(asset.source_object_id not in source_ids for asset in self.assets):
            raise ValueError("Asset lacks a source object identity")
        covered = {asset.source_object_id for asset in self.assets}
        required = {obj.id for obj in understanding.objects if obj.role != "visual_context"}
        if required - covered:
            raise ValueError(f"Missing task-relevant objects: {sorted(required - covered)}")
        for item in [*self.assets, *self.diagnostic_states]:
            if any(not 0 <= f < frame_count for f in item.evidence_frames):
                raise ValueError("Scene evidence is outside video")
            if observed_frames is not None and set(item.evidence_frames) - observed_frames:
                raise ValueError("Scene cites source frames that were not supplied")


class RestCheck(Contract):
    state_index: int = Field(ge=0)
    rationale: str


class SweepCheck(Contract):
    joint_id: Identifier
    state_index: int = Field(ge=0)
    end_position: float
    rationale: str


class ExcludedState(Contract):
    state_index: int = Field(ge=0)
    reason: str


class AssemblyCheck(Contract):
    state_index: int = Field(ge=0)
    mode: Literal["hold", "level_then_release"]
    lift_m: float = Field(default=0.12, ge=0.05, le=0.5)
    rationale: str


class SurfaceAsset(Contract):
    asset_id: Identifier
    slope_radii_m: list[Positive] = Field(min_length=2, max_length=2)


class SurfaceCheck(Contract):
    """Inspect the top of supported, approximately axial rigid objects."""

    assets: list[SurfaceAsset] = Field(min_length=1, max_length=16)
    rationale: str


class FreefallCheck(Contract):
    asset_id: Identifier
    support_asset_id: Identifier
    support_part_id: Identifier
    state_indices: list[int] = Field(min_length=1, max_length=12)
    rationale: str


class InterfaceCase(Contract):
    state_index: int = Field(ge=0)
    target_id: Identifier


class InterfaceCheck(Contract):
    """Hex-tool contact diagnostic; world-Z entry and local-Z ±15° turning."""

    tool_id: Identifier
    fixture_id: Identifier
    support_id: Identifier
    fastener_ids: list[Identifier] = Field(min_length=2, max_length=8)
    cases: list[InterfaceCase] = Field(min_length=1, max_length=16)
    tool_part_prefix: Identifier = "long_hex_"
    socket_wall_prefix: Identifier = "socket_wall_"
    socket_floor_prefix: Identifier = "head_floor_hex_"
    contact_offset_m: float = Field(default=0.000005, gt=0, le=0.0001)
    insertion_lift_m: float = Field(default=0.0036, gt=0, le=0.02)
    rationale: str


class ValidationPlan(Contract):
    rest_checks: list[RestCheck]
    sweeps: list[SweepCheck]
    excluded_states: list[ExcludedState]
    assembly_checks: list[AssemblyCheck] = Field(default_factory=list)
    surface_checks: list[SurfaceCheck] = Field(default_factory=list)
    freefall_checks: list[FreefallCheck] = Field(default_factory=list)
    interface_checks: list[InterfaceCheck] = Field(default_factory=list)
    additional_requirements: list[str]

    def validate_scene(self, scene: SceneSpec) -> None:
        if not any(
            (
                self.rest_checks,
                self.sweeps,
                self.assembly_checks,
                self.surface_checks,
                self.freefall_checks,
                self.interface_checks,
            )
        ):
            raise ValueError("Validation requires at least one executable physical diagnostic")
        count = len(scene.diagnostic_states)
        rest = [r.state_index for r in self.rest_checks]
        excluded = [r.state_index for r in self.excluded_states]
        if len(rest) != len(set(rest)) or len(excluded) != len(set(excluded)):
            raise ValueError("Repeated validation state")
        if set(rest) & set(excluded) or set(rest + excluded) != set(range(count)):
            raise ValueError("Every state must have either a rest check or an explicit exclusion")
        joints = {j.id: j for j in scene.joints}
        for sweep in self.sweeps:
            j = joints.get(sweep.joint_id)
            if j is None or j.type == "fixed" or j.drive is None or j.drive.max_force <= 0:
                raise ValueError("Sweep requires an explicitly driven movable joint")
            if not 0 <= sweep.state_index < count or not j.lower <= sweep.end_position <= j.upper:
                raise ValueError("Invalid sweep state or target")
        swept = {s.joint_id for s in self.sweeps}
        moving = {
            q.joint_id
            for state in scene.diagnostic_states
            for q in state.joint_positions
            if q.value != joints[q.joint_id].initial_position
        }
        if moving - swept:
            raise ValueError("Every source-moving joint needs a continuous sweep")
        if scene.attachments:
            if len({a.parent for a in scene.attachments}) != 1:
                raise ValueError("Current assembly probe requires one externally held parent")
            if {check.mode for check in self.assembly_checks} != {"hold", "level_then_release"}:
                raise ValueError("Assembly needs both hold and level-then-release diagnostics")
            children = {a.child for a in scene.attachments}
            if any(
                p.collision and p.shape != "box"
                for a in scene.assets
                if a.id in children
                for p in a.parts
            ):
                raise ValueError(
                    "Assembly release clearance currently requires box child colliders"
                )
            for check in self.assembly_checks:
                if not 0 <= check.state_index < count:
                    raise ValueError("Assembly check state is out of range")
                state = scene.diagnostic_states[check.state_index]
                if set(state.enabled_attachments) != {a.id for a in scene.attachments}:
                    raise ValueError("Assembly probe starts with all connections enabled")
        elif self.assembly_checks:
            raise ValueError("Assembly probes require explicit attachments")
        assets = {a.id: a for a in scene.assets}
        for check in self.interface_checks:
            role_ids = [check.tool_id, check.fixture_id, check.support_id, *check.fastener_ids]
            unique(role_ids, "interface role")
            if any(name not in assets for name in role_ids):
                raise ValueError("Unknown interface body")
            if assets[check.tool_id].body_type != "kinematic":
                raise ValueError("Interface diagnostic requires an explicitly externally held tool")
            if any(
                assets[name].body_type != "dynamic"
                for name in [check.fixture_id, *check.fastener_ids]
            ):
                raise ValueError("Fixture and fasteners must remain dynamic")
            if assets[check.support_id].body_type != "static":
                raise ValueError("Interface support must be static")
            if len({(c.state_index, c.target_id) for c in check.cases}) != len(check.cases):
                raise ValueError("Duplicate interface case")
            for case in check.cases:
                if case.target_id not in check.fastener_ids or not 0 <= case.state_index < count:
                    raise ValueError("Invalid interface target/state")
            for asset in scene.assets:
                for part in asset.parts:
                    if part.collision and (
                        part.shape not in {"box", "sphere", "cylinder"}
                        or (part.shape == "sphere" and max(part.size) - min(part.size) > 1e-8)
                        or (part.shape == "cylinder" and abs(part.size[0] - part.size[1]) > 1e-8)
                    ):
                        raise ValueError(
                            "Interface clearance audit requires native Cube/Sphere/Cylinder colliders"
                        )
            for name, prefixes in [
                (check.tool_id, [check.tool_part_prefix]),
                *[
                    (n, [check.socket_wall_prefix, check.socket_floor_prefix])
                    for n in check.fastener_ids
                ],
            ]:
                for prefix in prefixes:
                    parts = [
                        p for p in assets[name].parts if p.collision and p.id.startswith(prefix)
                    ]
                    if not parts:
                        raise ValueError("Interface geometry lacks declared contact parts")
                    if prefix != check.socket_floor_prefix and any(
                        p.contact_offset_m != check.contact_offset_m for p in parts
                    ):
                        raise ValueError(
                            "Interface contact offsets must match the delivered geometry"
                        )
        for check in self.surface_checks:
            unique([a.asset_id for a in check.assets], "surface asset")
            for target in check.assets:
                asset = assets.get(target.asset_id)
                if asset is None or asset.body_type != "dynamic":
                    raise ValueError("Surface contact tests require a dynamic asset")
                if target.slope_radii_m[0] >= target.slope_radii_m[1]:
                    raise ValueError("Surface slope radii must be strictly increasing")
                if any(
                    p.opacity > 0
                    and not (
                        p.shape in {"box", "cylinder", "lathe"}
                        or (p.shape == "sphere" and max(p.size) - min(p.size) > 1e-8)
                        or (
                            p.shape in {"cone", "capsule"}
                            and (
                                abs(p.size[0] - p.size[1]) > 1e-8
                                or (p.shape == "capsule" and p.size[2] < p.size[0])
                            )
                        )
                    )
                    for p in asset.parts
                ):
                    raise ValueError(
                        "Surface triangle audit supports compiled Cube/Cylinder/Mesh geometry"
                    )
        for check in self.freefall_checks:
            asset = assets.get(check.asset_id)
            support = assets.get(check.support_asset_id)
            if asset is None or asset.body_type != "dynamic":
                raise ValueError("Freefall requires a dynamic sphere")
            colliders = [p for p in asset.parts if p.collision]
            if (
                len(colliders) != 1
                or colliders[0].shape != "sphere"
                or max(colliders[0].size) - min(colliders[0].size) > 1e-8
            ):
                raise ValueError("Freefall audit requires one isotropic Sphere collider")
            if support is None or support.body_type != "static":
                raise ValueError("Freefall support must be static")
            part = next((p for p in support.parts if p.id == check.support_part_id), None)
            if part is None or part.shape != "box" or not part.collision:
                raise ValueError("Freefall support must be a colliding box part")
            if any(abs(x) > 1e-8 for x in [*support.rotation_deg, *part.rotation_deg]):
                raise ValueError("Analytic freefall support must be axis aligned")
            if len(set(check.state_indices)) != len(check.state_indices) or any(
                not 0 <= i < count for i in check.state_indices
            ):
                raise ValueError("Invalid freefall state indices")


class Proposal(Contract):
    scene: SceneSpec
    validation: ValidationPlan

    @model_validator(mode="after")
    def check_plan(self) -> Proposal:
        self.validation.validate_scene(self.scene)
        return self


class Finding(Contract):
    severity: Literal["high", "medium", "low"]
    category: Literal[
        "missing_object",
        "wrong_initial_state",
        "geometry",
        "action_understanding",
        "camera",
        "physics",
        "appearance",
    ]
    asset_id: str
    source_evidence_frames: list[int]
    render_evidence: str
    problem: str
    required_repair: str


class Completeness(Contract):
    later_revealed_objects_accounted_for: bool
    complete_action_sequence: bool
    initial_and_final_state_consistent: bool
    non_target_obstacles_accounted_for: bool


class Review(Contract):
    scene_id: Identifier
    round: int = Field(ge=1)
    verdict: Literal["repair", "acceptable_with_stated_uncertainties"]
    findings: list[Finding]
    completeness: Completeness
    remaining_uncertainties: list[str]
    acceptance_reasoning: str

    def validate_evidence(self, observed_frames: set[int]) -> None:
        if any(set(f.source_evidence_frames) - observed_frames for f in self.findings):
            raise ValueError("Review cites source frames that were not supplied")

    def accepted(self) -> bool:
        return (
            self.verdict == "acceptable_with_stated_uncertainties"
            and all(self.completeness.model_dump().values())
            and not any(f.severity in {"high", "medium"} for f in self.findings)
        )
