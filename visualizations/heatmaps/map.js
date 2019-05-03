const express = require('express');
const fs = require('fs');

const app = express();
const port = 3000;
app.set('view engine', 'ejs');
var mongoose = require('mongoose');

var mongoDB = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data';

mongoose.connect(mongoDB, { useNewUrlParser: true });
var db = mongoose.connection;
db.on('error', console.error.bind(console, 'MongoDB connection error:'));
sample_data = [];
arrest = [];

let caseSchema = new mongoose.Schema({
  "_id": String,
  "CaseNumber": String,
  "Location": String,
  "Reported Date": String,
  "Month": String,
  "Year": String,
  "Offense Desc": String,
  "Statute Code": String,
  "Statute Desc": String,
  "Counts": String,
  "Reporting Officer": String,
})
let Cases = mongoose.model('Cases', caseSchema);
app.get('/', function(req, res) {
  Cases.find({}, function (err, cases) {
    cases.forEach(function(el) {
      row = String(el).match(/\w+|'[^']+'/g);
      let latitude = parseInt(row[row.indexOf("latitude")+1])
      let longitude = parseInt(row[row.indexOf("longitude")+1])
      let latdecimal = row[row.indexOf("latitude")+2]
      let londecimal = row[row.indexOf("longitude")+2]
      let statute = row[row.indexOf("\'Statute Desc\'")+1]
      let caseDate = row[row.indexOf("\'Reported Date\'")+1]
      let address = row[row.indexOf("Location")+1]
      let arrestDate = row[row.indexOf("\'Arrest Date\'")+1]
      while (latdecimal > 1) {
        latdecimal /= 10
      }
      latitude += latdecimal
      while (londecimal > 1) {
        londecimal /= 10
      }
      longitude += londecimal
      longitude = (-1)*longitude
      let isArrest = false
      if (row[row.indexOf("Arrests")+1] == "_id") {
        isArrest = true
      }
      sample_data.push([parseFloat(latitude),parseFloat(longitude),isArrest,statute,caseDate,address,arrestDate])
    })
    render(res);
  })
});

function render(res) {
  res.render("test.ejs", {data: sample_data});
}

app.listen(port, () => console.log('Port 3000...'));
