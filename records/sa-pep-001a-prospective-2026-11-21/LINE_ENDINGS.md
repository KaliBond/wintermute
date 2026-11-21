# Line-ending note (added 2026-11-21 AEDT, at commit time)

The seven source files below were CRLF on Kari McKern's computer (Nitra). The commit path normalised them to LF, so their bytes in this repository differ from MANIFEST.txt by line endings only. The conversion is exactly reversible: converting each repository copy back to CRLF (for example `unix2dos`, or `b.replace(b'\n', b'\r\n')` in Python) reproduces the original bytes and the MANIFEST.txt SHA-256. All other files, including JUNO_Unified_Dataset_v3_2026-10-19.csv, are committed byte-identical.

| File | SHA-256 original (CRLF, as in MANIFEST.txt) | SHA-256 as committed (LF) |
|---|---|---|
| sources/Users__julie__wt-best__analysis__camnations5_drc_1960_2026.csv | 073c484d07d9e7e24e1c6ee04d79b9c6e0ba3992868b2d084e5a90ad636a5108 | 3f464f66785b24c062faf0129556f1279291f90b899b15137649e4a40533ac40 |
| sources/Users__julie__wt-best__analysis__camnations5_egypt_1952_2026.csv | ac24c5b9bb906b357b745bec5ce1892cd444ffe65f9b1741ff98b0f85866bb5d | 3aeb9e4be28af9e93eec9abc3a6575949a797a6196db96cd790a1703ef16c39f |
| sources/Users__julie__wt-best__analysis__camnations5_eritrea_1993_2026.csv | ca101dbf1b44e2d69819d23aa031614ee4cc8cf646a55148f25e5db07506858d | 049fe73a9326006ee1e0295472d883d264bf20bf30692296202b13a8cb7c5d0c |
| sources/Users__julie__wt-best__analysis__camnations5_ethiopia_1960_2026.csv | 0fb5a1a51630bf9610f2a4a20c4ed67c6085c155b5b20443a0035e8692e45664 | 6f32c212f9f5fe92a75e67b9f68641a064488194f0d04ba4fb31359d570128a2 |
| sources/Users__julie__wt-best__analysis__camnations5_mauritania_1960_2026.csv | 23a78ec0462d209a2556a2e758de8539b302df07ba26a170a8fda0bc4d086eec | 0cf8c5c76babb7cf5ae3a3246521055b5f80096405a0526457a37ec8e6cd8393 |
| sources/Users__julie__wt-best__analysis__camnations5_nigeria_1960_2026.csv | 3cb7f0826d564e48c089491bbbbd78b05762d2e7d2fb50b820c876d7375385af | fbd97d578b66b81f0383a98da41ed539454238dbc9a3cf97ecace642471b2800 |
| sources/Users__julie__wt-best__juno__Vietnam_CAMS_mfin_1965-2026_block1.csv | fbca3810d0dcc94ac19cee17dc1135a4ce12a706fd575ae4d540507be63e388f | a70d02d80abe4b1bd457e0e473105deae3f09be8c92de911880f582f01611b20 |
