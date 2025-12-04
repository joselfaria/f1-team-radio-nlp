from urllib.request import urlopen
import json
import requests
import csv
import tempfile
import whisper


YEARS = [2025, 2024]

session_keys = []

print("\nCOLETANDO SESSION KEYS DOS ANOS:", YEARS)

for year in YEARS:
    try:
        URL = f"https://api.openf1.org/v1/sessions?year={year}"
        resp = urlopen(URL)
        data = json.loads(resp.read().decode("utf-8"))

        keys = {s["session_key"] for s in data}
        session_keys.extend(keys)

        print(f" Ano {year}: {len(keys)} sessões")

    except Exception as e:
        print(f"Erro ao coletar sessions de {year}: {e}")

session_keys = sorted(list(set(session_keys)))  # remover duplicados

print("\nTotal de sessões combinadas:", len(session_keys))

session_info_cache = {}

print("\nCOLETANDO METADADOS DE SESSÕES")

for sk in session_keys:
    try:
        url = f"https://api.openf1.org/v1/sessions?session_key={sk}"
        resp = urlopen(url)
        data = json.loads(resp.read().decode("utf-8"))
        if len(data) > 0:
            s = data[0]
            session_info_cache[sk] = {
                "session_type": s.get("session_type"),
                "circuit_short_name": s.get("circuit_short_name")
            }
    except Exception as e:
        print(f"Erro ao coletar metadados da sessão {sk}: {e}")


print("\nCOLETANDO PILOTOS DE TODAS AS SESSÕES")

all_drivers = set()
driver_info_cache = {}   # salvar nome + equipe

for sk in session_keys:
    try:
        url = f"https://api.openf1.org/v1/drivers?session_key={sk}"
        resp = urlopen(url)
        data = json.loads(resp.read().decode("utf-8"))

        for d in data:
            num = d.get("driver_number")
            if num is None:
                continue
    
            all_drivers.add(num)

            # salva dados do piloto
            driver_info_cache[num] = {
                "driver_name": f"{d.get('first_name','')} {d.get('last_name','')}".strip(),
                "team_name": d.get("team_name")
            }

    except Exception as e:
        print(f"Erro ao coletar pilotos da session_key {sk}: {e}")

DRIVER_NUMBERS = tuple(sorted(all_drivers))

print("\nTotal de pilotos encontrados:", len(DRIVER_NUMBERS))
print("Pilotos:")
print(DRIVER_NUMBERS)


print("\n INICIANDO WHISPER")

CSV_OUTPUT = "data/raw/transcricoes_radio.csv"

model = whisper.load_model("base")


def get_team_radio(session_key, driver_number):
    url = f"https://api.openf1.org/v1/team_radio?session_key={session_key}&driver_number={driver_number}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Falha ao acessar rádio do piloto {driver_number} na sessão {session_key}")
            return []
        data = response.json()
        return data if isinstance(data, list) else []
    except:
        return []


with open(CSV_OUTPUT, mode="w", newline="", encoding="utf-8", buffering=1) as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow([
        "session_key",
        "session_type",
        "circuit",
        "driver_number",
        "driver_name",
        "team_name",
        "date",
        "transcription"
    ])

    for session in session_keys:
        print(f"\nProcessando sessão {session}")

        session_meta = session_info_cache.get(session, {})
        session_type = session_meta.get("session_type")
        circuit = session_meta.get("circuit_short_name")

        for driver in DRIVER_NUMBERS:

            driver_meta = driver_info_cache.get(driver, {})
            driver_name = driver_meta.get("driver_name")
            team_name = driver_meta.get("team_name")

            team_radio = get_team_radio(session, driver)

            if len(team_radio) == 0:
                print(f" Sem dados para piloto {driver} na sessão {session}...")
                continue

            print(f" Piloto {driver}: {len(team_radio)} mensagens")

            for msg in team_radio:
                date = msg.get("date")
                recording_url = msg.get("recording_url")

                if not recording_url:
                    continue

                try:
                    audio_response = requests.get(recording_url)
                    if audio_response.status_code != 200:
                        print("   - Falha ao baixar áudio...")
                        continue
                except:
                    continue

                # salvar audio temporário
                with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_file:
                    tmp_file.write(audio_response.content)
                    tmp_file.flush()

                    try:
                        result = model.transcribe(tmp_file.name, language="en")
                        text = result["text"].strip()
                    except Exception as e:
                        print("   Erro na transcrição:", e)
                        continue

                writer.writerow([
                    session,
                    session_type,
                    circuit,
                    driver,
                    driver_name,
                    team_name,
                    date,
                    text
                ])

                csv_file.flush()
                print(f"   - OK: {date}")

print("\nPROCESSO FINALIZADO")
