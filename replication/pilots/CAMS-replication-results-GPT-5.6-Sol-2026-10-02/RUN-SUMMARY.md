# CAMS Replication Pilot — ChatGPT / GPT-5.6 Sol

Date: 2026-10-02 (Australia/Sydney)
Input: CAMS-Replication-Kit-v1.0.zip

## Results

| Entity | Year | Node | Coherence | Capacity | Stress | Abstraction |
|---|---:|---|---:|---:|---:|---:|
| BLIND-A | 100 BCE | Helm | 8 | 8 | 8 | 8 |
| BLIND-B | 100 CE | Helm | 8 | 8 | 6 | 8 |

## Validity status

These are **pilot/diagnostic scores, not valid TP-1 third-party replication draws**. The governing protocol requires one packet per fresh session and prohibits a scorer session from seeing both packets. This run occurred in one ChatGPT conversation, so session isolation cannot be certified. No Run A scores were supplied or used.

The results should therefore **not** be included in `ensemble-xp.csv` or `envelope-xp.csv` as valid foreign draws. They can be retained separately as a platform pilot.

## Packet-integrity finding

`PACKET-HASHES.txt` lists SHA-256 values for the full packets. The bytewise hashes of the ZIP-extracted full files differ because the ZIP copies use CRLF line endings. After normalising CRLF to LF, both full-packet hashes match the listed values exactly:

- BLIND-A full, LF-normalised: `5e501e8247773ec7094218e23180f4eb5f7c15298dc8c6c86cd1cc1786a7b15b`
- BLIND-B full, LF-normalised: `e4806f50a229c358305c48634e4dfb5e76746ce744d15568901506fc45c290ac`

The protocol requires use of redacted packets, but `PACKET-HASHES.txt` does not provide hashes for those redacted files. Their computed LF-normalised hashes are recorded in `packet-hashes-computed.txt`.

## Recommended treatment

Keep these scores in a separate provenance block. For a protocol-valid replication, obtain at least three independent draws per packet from at least two distinct model lineages or human experts, with each scorer session exposed to only one redacted packet.
