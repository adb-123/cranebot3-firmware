import asyncio
import unittest
from types import SimpleNamespace

from nf_robot.generated.nf import common, control
from nf_robot.host.observer import AsyncObserver


class _FakeClient:
    def __init__(self):
        self.commands = []

    async def send_commands(self, commands):
        self.commands.append(commands)


def _observer(anchor_type):
    observer = AsyncObserver.__new__(AsyncObserver)
    observer.config = SimpleNamespace(anchor_type=anchor_type)
    observer.config_path = None
    observer.anchors = {}
    observer.gripper_client = None
    return observer


class TestObserverCommandRouting(unittest.IsolatedAsyncioTestCase):
    async def test_gripper_jog_speed_routes_to_aim_speed(self):
        observer = _observer(common.AnchorType.PILOT)
        observer.gripper_client = _FakeClient()

        await observer._handle_jog_spool(
            control.JogSpool(is_gripper=True, speed=0.25)
        )
        await asyncio.sleep(0)

        self.assertEqual(observer.gripper_client.commands, [{'aim_speed': 0.25}])

    async def test_gripper_jog_offset_routes_to_jog(self):
        observer = _observer(common.AnchorType.PILOT)
        observer.gripper_client = _FakeClient()

        await observer._handle_jog_spool(
            control.JogSpool(is_gripper=True, offset=0.04)
        )
        await asyncio.sleep(0)

        self.assertEqual(observer.gripper_client.commands, [{'jog': 0.04}])

    async def test_arp_tighten_routes_spool_num(self):
        observer = _observer(common.AnchorType.ARPEGGIO)
        observer.anchors[1] = _FakeClient()

        await observer._handle_single_component_action(
            control.SingleComponentAction(
                is_gripper=False,
                anchor_num=1,
                action=control.ComponentAction.TIGHTEN,
                spool_num=0,
            )
        )

        self.assertEqual(observer.anchors[1].commands, [{'tighten': 0}])

    async def test_arp_relax_routes_spool_num(self):
        observer = _observer(common.AnchorType.ARPEGGIO)
        observer.anchors[1] = _FakeClient()

        await observer._handle_single_component_action(
            control.SingleComponentAction(
                is_gripper=False,
                anchor_num=1,
                action=control.ComponentAction.RELAX,
                spool_num=1,
            )
        )

        self.assertEqual(observer.anchors[1].commands, [{'relax': 1}])

    async def test_arp_stow_routes_spool_num(self):
        observer = _observer(common.AnchorType.ARPEGGIO)
        observer.anchors[1] = _FakeClient()

        await observer._handle_single_component_action(
            control.SingleComponentAction(
                is_gripper=False,
                anchor_num=1,
                action=control.ComponentAction.STOW,
                spool_num=1,
            )
        )

        self.assertEqual(observer.anchors[1].commands, [{'stow': 1}])

    async def test_arp_tighten_without_spool_num_does_not_send_invalid_none(self):
        observer = _observer(common.AnchorType.ARPEGGIO)
        observer.anchors[1] = _FakeClient()

        await observer._handle_single_component_action(
            control.SingleComponentAction(
                is_gripper=False,
                anchor_num=1,
                action=control.ComponentAction.TIGHTEN,
            )
        )

        self.assertEqual(observer.anchors[1].commands, [])

    async def test_pilot_tighten_keeps_legacy_none_payload(self):
        observer = _observer(common.AnchorType.PILOT)
        observer.anchors[2] = _FakeClient()

        await observer._handle_single_component_action(
            control.SingleComponentAction(
                is_gripper=False,
                anchor_num=2,
                action=control.ComponentAction.TIGHTEN,
            )
        )

        self.assertEqual(observer.anchors[2].commands, [{'tighten': None}])
