import sys; sys.path.insert(0,'/home/user/yara')
from node3_matchday_board import build_ufc_card

ufc_data = {
    "gw": 36, "league": "Premier League",
    "matchday": "Saturday 3 May",
    "fixture_label": "Arsenal vs Liverpool",
    "badge_type": "TOP-6 CLASH",
    "p1": {
        "first_name":"Bukayo","last_name":"Saka",
        "club":"Arsenal","position":"MID",
        "risk_pct":82,"archetype":"FRAGILE",
        "image_path": "/home/user/yara/output/saka.png",
        "gladiator_stat": "Has never completed 90' against a Top-6 side this season.",
        "vitals": {
            "acwr":"1.42","strain":"87","recurrence":"34%",
            "load_slope":"+8%","fatigue":"74",
            "goals_per_game":"0.52","assists_per_game":"0.31","form":"7.8",
        },
    },
    "p2": {
        "first_name":"Mohamed","last_name":"Salah",
        "club":"Liverpool","position":"MID",
        "risk_pct":78,"archetype":"FRAGILE",
        "image_path": "/home/user/yara/output/salah.png",
        "gladiator_stat": "Subbed off in 3 of last 4. Returns 18% over seasonal load.",
        "vitals": {
            "acwr":"1.18","strain":"72","recurrence":"19%",
            "load_slope":"+18%","fatigue":"61",
            "goals_per_game":"0.71","assists_per_game":"0.19","form":"8.4",
        },
    },
    "verdict": ""
}
build_ufc_card(ufc_data, output_path="/home/user/yara/output/ufc_battle_card.png")
print("Done.")