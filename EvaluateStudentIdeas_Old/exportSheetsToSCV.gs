function exportSheetsToCSV() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheets = ss.getSheets();
  const folder = DriveApp.createFolder(ss.getName() + "_CSV_Export_" + new Date().toISOString());

  sheets.forEach(sheet => {
    const name = sheet.getName();
    if (name === "Summary") return; // skip Summary if you want
    
    const data = sheet.getDataRange().getValues();
    let csv = data.map(row => row.map(item => {
      if (typeof item === "string") {
        // escape quotes
        return '"' + item.replace(/"/g, '""') + '"';
      }
      return item;
    }).join(",")).join("\n");

    const file = folder.createFile(name + ".csv", csv, MimeType.CSV);
    Logger.log("Exported: " + file.getUrl());
  });

  SpreadsheetApp.getUi().alert("Export complete! Check Google Drive folder: " + folder.getName());
}
