stocks = [
  "AI.PA",
  "AIR.PA",
  "ALO.PA",
  "MT.PA",
  "ATO.PA",
  "CS.PA",
  "BNP.PA",
  "EN.PA",
  "CAP.PA",
  "CA.PA",
  "ACA.PA",
  "BN.PA",
  "KER.PA",
  "ORA.PA",
  "LR.PA",
  "OR.PA",
  "MC.PA",
  "RI.PA",
  "PUB.PA",
  "RNO.PA",
  "SAF.PA",
  "SGO.PA",
  "SAN.PA",
  "SU.PA",
  "GLE.PA",
  "SW.PA",
  "STMPA.PA",
  "HO.PA",
  "TTE.PA",
  "VIE.PA",
  "DG.PA",
  "VIV.PA",
  "FR.PA",
]

from PIL import Image

for stock in stocks:
    new_image = Image.new("RGB", (100, 100))
    file_name = f"plot1_{stock}.png"
    new_image.save(file_name, "PNG")
    new_image.close()

for stock in stocks:
    new_image = Image.new("RGB", (100, 100))
    file_name = f"plot2_{stock}.png"
    new_image.save(file_name, "PNG")
    new_image.close()
