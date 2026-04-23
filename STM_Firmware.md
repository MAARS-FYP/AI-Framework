# RF Front-End Controller Firmware

STM32 NUCLEO-L476RG firmware for controlling and monitoring an RF front-end consisting of LNA power switches, RF signal-path switches, an IF variable-gain amplifier (LMH6401), and power-level ADC monitors.

---

## Hardware Pin Map

### 3 V LNA Power Switch

| Signal | Pin | Direction | Description                         |
| ------ | --- | --------- | ----------------------------------- |
| ON     | PC3 | Output    | Drive **High** to enable power      |
| FAULT  | PC2 | Input     | Reads **Low** when fault is present |

### 5 V LNA Power Switch

| Signal | Pin  | Direction | Description                         |
| ------ | ---- | --------- | ----------------------------------- |
| ON     | PB7  | Output    | Drive **High** to enable power      |
| FAULT  | PC13 | Input     | Reads **Low** when fault is present |

### 3-Way RF Switch (A = PA4, B = PB0 — pulled down externally)

| A    | B    | RF COM routed to       |
| ---- | ---- | ---------------------- |
| Low  | Low  | RF1 — 20 MHz BW filter |
| High | Low  | RF2 — 10 MHz BW filter |
| Low  | High | RF3 — 1 MHz BW filter  |
| High | High | All OFF                |

### 4-Way RF Switch (A = PC0, B = PC1 — pulled down externally)

| A    | B    | RF COM routed to        |
| ---- | ---- | ----------------------- |
| Low  | Low  | RF1 — External IF input |
| High | Low  | RF2 — 1 MHz filter      |
| Low  | High | RF3 — 10 MHz BW filter  |
| High | High | RF4 — 20 MHz BW filter  |

### IF Amplifier — LMH6401 (SPI3 via 3.3 V → 1.8 V level shifter)

| Signal | Pin  | Notes                                                         |
| ------ | ---- | ------------------------------------------------------------- |
| OE     | PA15 | Level-shifter output-enable, active-low (drive Low to enable) |
| SCK    | PC10 | SPI3 clock — 1.25 MHz                                         |
| MOSI   | PC12 | SPI3 data to IC                                               |
| MISO   | PC11 | SPI3 data from IC                                             |
| CS     | PD2  | Chip select, active-low                                       |

### ADC Power Level Monitors

| Channel  | Pin | Measures                  |
| -------- | --- | ------------------------- |
| Post-LNA | PA0 | Power level after LNA     |
| Pre-IF   | PA1 | Power level before IF amp |

ADC resolution: 12-bit (0 – 4095 counts). Voltage conversion logic to be added.

### UART (Serial CLI)

- **Port:** USART2 via ST-Link USB VCP
- **Baud:** 115200, 8N1, no flow control

---

## LMH6401 SPI Register Map

| Address | R/W | Register                      | Default |
| ------- | --- | ----------------------------- | ------- |
| 0x00    | R   | Revision ID                   | 0x03    |
| 0x01    | R   | Product ID                    | 0x00    |
| 0x02    | R/W | Gain Control                  | 0x20    |
| 0x03    | R/W | Reserved (always write 0x8C)  | 0x8C    |
| 0x04    | R/W | Thermal Feedback Gain Control | 0x27    |
| 0x05    | R/W | Thermal Feedback Freq Control | 0x45    |

### Gain Control Register (0x02)

| Bit   | Field       | Description                |
| ----- | ----------- | -------------------------- |
| 7     | Reserved    | Always write 0             |
| 6     | Power Down  | 0 = Active, 1 = Power Down |
| 5 : 0 | Attenuation | See gain table below       |

### Gain / Attenuation Table (Table 10)

| Attenuation (dB) | Gain (dB) | Register (0x02) |
| ---------------- | --------- | --------------- |
| 0                | **+26**   | 0x00            |
| 1                | 25        | 0x01            |
| 5                | 21        | 0x05            |
| 10               | 16        | 0x0A            |
| 15               | 11        | 0x0F            |
| 20               | 6         | 0x14            |
| 26               | 0         | 0x1A            |
| 31               | −5        | 0x1F            |
| **32**           | **−6**    | 0x20 _(hw min)_ |

SPI frame: 16 bits MSB-first, CPOL=0 CPHA=0.  
Write: `[R/W=0][A6:A0][D7:D0]`  
Read: `[R/W=1][A6:A0][dummy]` — IC drives MISO on clocks 9–16.

---

## Serial CLI Reference

Connect to the ST-Link VCP at **115200 baud, 8N1**.  
Commands are terminated by **Enter** (CR or LF). Backspace is supported.

### High-Level Host Commands (Recommended for AI Host Integration)

These commands are intended for the AI host application. The host sends only agent outputs, and firmware handles the low-level hardware mapping internally.

| Command     | Action                                                                                    |
| ----------- | ----------------------------------------------------------------------------------------- |
| `lna 3`     | Set LNA supply selection to 3 V (firmware enforces rail exclusivity)                      |
| `lna 5`     | Set LNA supply selection to 5 V (firmware enforces rail exclusivity)                      |
| `filter 1`  | Select 1 MHz filter path (firmware configures both RF muxes)                              |
| `filter 10` | Select 10 MHz filter path (firmware configures both RF muxes)                             |
| `filter 20` | Select 20 MHz filter path (firmware configures both RF muxes)                             |
| `ifamp <x>` | Apply IF amp setting from agent value `x` (firmware performs internal conversion/mapping) |

> Low-level commands below remain available for manual debug and bring-up.

### Power Switches

| Command      | Action                                             |
| ------------ | -------------------------------------------------- |
| `pwr3 on`    | Enable 3 V LNA power switch (PC3 High)             |
| `pwr3 off`   | Disable 3 V LNA power switch (PC3 Low)             |
| `pwr5 on`    | Enable 5 V LNA power switch (PB7 High)             |
| `pwr5 off`   | Disable 5 V LNA power switch (PB7 Low)             |
| `pwr status` | Print ON/OFF state and fault pin for both switches |

### 3-Way RF Switch

| Command      | Action                             |
| ------------ | ---------------------------------- |
| `rf3 1`      | Route COM → RF1 (20 MHz BW filter) |
| `rf3 2`      | Route COM → RF2 (10 MHz BW filter) |
| `rf3 3`      | Route COM → RF3 (1 MHz BW filter)  |
| `rf3 off`    | All paths off (A=High, B=High)     |
| `rf3 status` | Print current path                 |

### 4-Way RF Switch

| Command      | Action                              |
| ------------ | ----------------------------------- |
| `rf4 1`      | Route COM → RF1 (External IF input) |
| `rf4 2`      | Route COM → RF2 (1 MHz filter)      |
| `rf4 3`      | Route COM → RF3 (10 MHz BW filter)  |
| `rf4 4`      | Route COM → RF4 (20 MHz BW filter)  |
| `rf4 status` | Print current path                  |

### IF Amplifier (LMH6401)

| Command            | Action                                             |
| ------------------ | -------------------------------------------------- |
| `ifamp att <0-32>` | Set attenuation in dB (0 = 26 dB gain, 32 = −6 dB) |
| `ifamp pwrdn on`   | Power down the LMH6401                             |
| `ifamp pwrdn off`  | Wake up the LMH6401 (active mode)                  |
| `ifamp status`     | Print revision ID, product ID, gain, power state   |

### ADC Power Monitors

| Command    | Action                                                              |
| ---------- | ------------------------------------------------------------------- |
| `adc read` | Print raw 12-bit ADC values for PA0 (post-LNA) and PA1 (pre-IF amp) |

### General

| Command  | Action                                  |
| -------- | --------------------------------------- |
| `status` | Print complete status of all subsystems |
| `help`   | Print command reference                 |

---

## Build & Flash

### Prerequisites

- STM32CubeIDE (provides the arm-none-eabi toolchain, CMake, and Ninja)
- STM32CubeProgrammer

### Build (VS Code task)

Press **Ctrl+Shift+B** or run the **Build STM32** task.  
Uses: `C:/Users/Danidu Dabare/AppData/Local/stm32cube/bundles/cmake/4.0.1+st.3/bin/cmake.exe`

### Flash (VS Code task)

Run the **Flash STM32** task (or **Build and Flash** to do both sequentially).  
Uses STM32CubeProgrammer CLI over SWD.

### Manual build (terminal)

```powershell
& "C:/Users/Danidu Dabare/AppData/Local/stm32cube/bundles/cmake/4.0.1+st.3/bin/cmake.exe" --build build/Debug
```

> **Note:** Do not use the system `cmake` (MSYS64) for building — it was built with a different version than what configured the project and will fail.

---

## Project File Structure

```
STM32_VSC_Blink/
├── Core/
│   ├── Inc/
│   │   ├── board_config.h      # All pin/peripheral definitions
│   │   ├── power_switch.h      # 3V/5V LNA power switch API
│   │   ├── rf_switch.h         # 3-way and 4-way RF switch API
│   │   ├── if_amp.h            # LMH6401 SPI driver API
│   │   ├── adc_reader.h        # ADC power monitor API
│   │   ├── cli.h               # Serial CLI API
│   │   ├── main.h
│   │   ├── stm32l4xx_hal_conf.h
│   │   └── stm32l4xx_it.h
│   └── Src/
│       ├── main.c              # Entry point, peripheral init, main loop
│       ├── power_switch.c      # LNA power switch control
│       ├── rf_switch.c         # RF switch path control
│       ├── if_amp.c            # LMH6401 SPI driver
│       ├── adc_reader.c        # ADC1 reader (PA0, PA1)
│       ├── cli.c               # UART command-line interface
│       ├── stm32l4xx_hal_msp.c
│       ├── stm32l4xx_it.c
│       ├── syscalls.c
│       ├── sysmem.c
│       └── system_stm32l4xx.c
├── Drivers/                    # STM32 HAL + CMSIS drivers
├── cmake/
│   └── stm32cubemx/
│       └── CMakeLists.txt      # Source and driver lists
├── .vscode/
│   ├── tasks.json              # Build and Flash tasks
│   └── launch.json             # Debug configuration
├── CMakeLists.txt
├── CMakePresets.json
├── STM32L476XX_FLASH.ld        # Linker script
└── startup_stm32l476xx.s       # Startup assembly
```
