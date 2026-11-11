#!/usr/bin/env python3
"""
Generate synthetic SAML-D test dataset with star and chain structures
up to 3rd hop, with increased transaction amounts for stress testing.

Chain structures (A→B→C→D→E) - from any node you can see up to 3 hops.
Star structures (hub H → many leaves) - coexisting with chains.
Cross-links between chains and stars for complex topology.
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

BASE = datetime(2022, 10, 7, 10, 35, 19)
MIN_AMOUNT = 5000.0  # Increased minimum amount

def fmt_time(dt): return dt.strftime("%H:%M:%S")
def fmt_date(dt): return dt.strftime("%Y-%m-%d")

rows = []
tx_counter = 0

def add_tx(sender, receiver, t, amount=None, currency_pair=("UK pounds","UK pounds"),
           locations=("UK","UK"), ptype="ACH", laundering=0, ltype="Normal"):
    global tx_counter
    tx_counter += 1
    if amount is None:
        amount = round(random.uniform(MIN_AMOUNT, 50000), 2)
    rows.append({
        "Time": fmt_time(t),
        "Date": fmt_date(t),
        "Sender_account": sender,
        "Receiver_account": receiver,
        "Amount": amount,
        "Payment_currency": currency_pair[0],
        "Received_currency": currency_pair[1],
        "Sender_bank_location": locations[0],
        "Receiver_bank_location": locations[1],
        "Payment_type": ptype,
        "Is_laundering": laundering,
        "Laundering_type": ltype,
    })

# ============================================================
# CHAIN STRUCTURES (3+ hops)
# ============================================================

# Chain 1: C1_1 → C1_2 → C1_3 → C1_4 → C1_5 → C1_6 (5 hops)
chain1 = ["C1_1", "C1_2", "C1_3", "C1_4", "C1_5", "C1_6"]
for i in range(len(chain1) - 1):
    t = BASE + timedelta(seconds=i * 30)
    add_tx(chain1[i], chain1[i+1], t, ptype="ACH")

# Chain 2: C2_1 → C2_2 → C2_3 → C2_4 → C2_5 → C2_6 (5 hops)
chain2 = ["C2_1", "C2_2", "C2_3", "C2_4", "C2_5", "C2_6"]
for i in range(len(chain2) - 1):
    t = BASE + timedelta(seconds=i * 30 + 300)
    add_tx(chain2[i], chain2[i+1], t, currency_pair=("UK pounds","Dirham"),
           locations=("UK","UAE"), ptype="Cross-border")

# Chain 3: C3_1 → C3_2 → C3_3 → C3_4 → C3_5 (4 hops)
chain3 = ["C3_1", "C3_2", "C3_3", "C3_4", "C3_5"]
for i in range(len(chain3) - 1):
    t = BASE + timedelta(seconds=i * 30 + 600)
    add_tx(chain3[i], chain3[i+1], t, currency_pair=("Dirham","UK pounds"),
           locations=("UAE","UK"), ptype="Cheque")

# Chain 4: C4_1 → C4_2 → C4_3 (2 hops)
chain4 = ["C4_1", "C4_2", "C4_3"]
for i in range(len(chain4) - 1):
    t = BASE + timedelta(seconds=i * 30 + 900)
    add_tx(chain4[i], chain4[i+1], t, ptype="Cash Deposit")

# ============================================================
# STAR STRUCTURES (hub → many leaves)
# ============================================================

# Star H1: H1 → 10 leaves
hub_h1 = "H1"
for i in range(1, 11):
    leaf = f"H1_L{i}"
    t = BASE + timedelta(seconds=1500 + i * 10)
    add_tx(hub_h1, leaf, t, currency_pair=("UK pounds","UK pounds"),
           locations=("UK","UK"), ptype="Cheque")

# Star H2: H2 → 8 leaves (H2 also in chain 2)
hub_h2 = "H2"
for i in range(1, 9):
    leaf = f"H2_L{i}"
    t = BASE + timedelta(seconds=1700 + i * 10)
    add_tx(hub_h2, leaf, t, currency_pair=("Dirham","UK pounds"),
           locations=("UAE","UK"), ptype="Cross-border")

# Star H3: H3 → 12 leaves
hub_h3 = "H3"
for i in range(1, 13):
    leaf = f"H3_L{i}"
    t = BASE + timedelta(seconds=1900 + i * 10)
    add_tx(hub_h3, leaf, t, currency_pair=("UK pounds","Dirham"),
           locations=("UK","UAE"), ptype="ACH")

# Star H4: H4 → 5 leaves
hub_h4 = "H4"
for i in range(1, 6):
    leaf = f"H4_L{i}"
    t = BASE + timedelta(seconds=2100 + i * 10)
    add_tx(hub_h4, leaf, t, currency_pair=("Dirham","Dirham"),
           locations=("UAE","UAE"), ptype="Cash Deposit")

# ============================================================
# CROSS-LINKS (star hubs ↔ chain accounts)
# ============================================================

# H1 → C1_3 (star hub to chain account)
t = BASE + timedelta(seconds=900)
add_tx(hub_h1, "C1_3", t, amount=25000.0, currency_pair=("UK pounds","UK pounds"),
       locations=("UK","UK"), ptype="Cross-border")

# C2_3 → H1 (chain account to star hub)
t = BASE + timedelta(seconds=930)
add_tx("C2_3", hub_h1, t, amount=25000.0, currency_pair=("Dirham","UK pounds"),
       locations=("UAE","UK"), ptype="Cross-border")

# H2 → C3_2 (star hub to chain account)
t = BASE + timedelta(seconds=960)
add_tx(hub_h2, "C3_2", t, amount=30000.0, currency_pair=("Dirham","UK pounds"),
       locations=("UAE","UK"), ptype="Cross-border")

# C3_4 → H3 (chain account to star hub)
t = BASE + timedelta(seconds=990)
add_tx("C3_4", hub_h3, t, amount=30000.0, currency_pair=("UK pounds","Dirham"),
       locations=("UK","UAE"), ptype="Cross-border")

# H4 → C4_2 (star hub to chain account)
t = BASE + timedelta(seconds=1020)
add_tx(hub_h4, "C4_2", t, amount=15000.0, currency_pair=("Dirham","Dirham"),
       locations=("UAE","UAE"), ptype="Cash Deposit")

# C4_3 → H4 (chain account to star hub)
t = BASE + timedelta(seconds=1050)
add_tx("C4_3", hub_h4, t, amount=15000.0, currency_pair=("Dirham","Dirham"),
       locations=("UAE","UAE"), ptype="Cash Deposit")

# ============================================================
# ADDITIONAL VARIATIONS - more complex topologies
# ============================================================

# Bidirectional chain pair: C5_1 ↔ C5_2 (both directions)
t = BASE + timedelta(seconds=2400)
add_tx("C5_1", "C5_2", t, amount=5000.0, ptype="ACH")
t = BASE + timedelta(seconds=2410)
add_tx("C5_2", "C5_1", t, amount=5000.0, ptype="ACH")

# Triangle: C6_1 → C6_2 → C6_3 → C6_1
t = BASE + timedelta(seconds=2500)
add_tx("C6_1", "C6_2", t, amount=5000.0, ptype="ACH")
t = BASE + timedelta(seconds=2510)
add_tx("C6_2", "C6_3", t, amount=5000.0, ptype="ACH")
t = BASE + timedelta(seconds=2520)
add_tx("C6_3", "C6_1", t, amount=5000.0, ptype="ACH")

# Fan-in to hub: many senders → H1
fan_in_senders = [f"FI_S{i}" for i in range(1, 6)]
for sender in fan_in_senders:
    t = BASE + timedelta(seconds=2600 + fan_in_senders.index(sender) * 10)
    add_tx(sender, hub_h1, t, amount=5000.0, ptype="ACH")

# Fan-out from hub: H1 → many receivers  
fan_out_receivers = [f"FO_R{i}" for i in range(1, 6)]
for receiver in fan_out_receivers:
    t = BASE + timedelta(seconds=2700 + fan_out_receivers.index(receiver) * 10)
    add_tx(hub_h1, receiver, t, amount=5000.0, ptype="ACH")

# Long chain for k=7 test: LC_1 → LC_2 → ... → LC_8 (7 hops)
long_chain = [f"LC_{i}" for i in range(1, 9)]
for i in range(len(long_chain) - 1):
    t = BASE + timedelta(seconds=2800 + i * 10)
    add_tx(long_chain[i], long_chain[i+1], t, amount=5000.0, ptype="ACH")

# Sort by timestamp
rows.sort(key=lambda r: r["Date"] + " " + r["Time"])

output_path = "/Users/patrickabainza/Documents/October_2026_projects/abacus_mvp_v10/test_data/saml_d_test_chain_star.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Time", "Date", "Sender_account", "Receiver_account", "Amount",
        "Payment_currency", "Received_currency", "Sender_bank_location",
        "Receiver_bank_location", "Payment_type", "Is_laundering", "Laundering_type"
    ])
    writer.writeheader()
    writer.writerows(rows)

# Analysis
all_senders = set(r["Sender_account"] for r in rows)
all_receivers = set(r["Receiver_account"] for r in rows)
chain_accts = sorted(all_senders & all_receivers)
amounts = [r["Amount"] for r in rows]

print(f"Generated {len(rows)} transactions in {output_path}")
print(f"Distinct accounts: {len(all_senders | all_receivers)}")
print(f"Chain accounts (both sender & receiver): {len(chain_accts)} -> {chain_accts}")
print(f"Amount range: ${min(amounts):,.2f} - ${max(amounts):,.2f}")
print(f"Average amount: ${sum(amounts)/len(amounts):,.2f}")
print(f"\nExpected multi-hop from C1_1 (k=3): h0=C1_1→C1_2, h1=C1_2→C1_3, h2=C1_3→C1_4, h3=C1_4→C1_5 + C1_4→C1_6")
print(f"Expected multi-hop from LC_1 (k=7): all 7 hops of long chain")
