import time
from collections import defaultdict


class ThreatMonitor:
    def __init__(self, threat_time=1.5, critical_time=3.5, clear_time=2):
        self.objects = defaultdict(
            lambda: {"first_seen": None, "last_seen": None, "state": "OBSERVED"}
        )
        self.threat_time = threat_time
        self.critical_time = critical_time
        self.clear_time = clear_time

    def update(self, track_id):
        now = time.time()
        obj = self.objects[track_id]

        if obj["first_seen"] is None:
            obj["first_seen"] = now

        obj["last_seen"] = now
        duration = now - obj["first_seen"]

        if duration >= self.critical_time:
            obj["state"] = "CRITICAL"
        elif duration >= self.threat_time:
            obj["state"] = "THREAT"
        else:
            obj["state"] = "OBSERVED"

        return obj["state"]

    def cleanup(self):
        now = time.time()
        for tid, obj in list(self.objects.items()):
            if now - obj["last_seen"] > self.clear_time:
                del self.objects[tid]
