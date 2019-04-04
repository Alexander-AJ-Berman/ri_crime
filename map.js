const express = require('express');
const fs = require('fs');

const app = express();
const port = 3002;
app.set('view engine', 'ejs');
var mongoose = require('mongoose');

var mongoDB = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data';

mongoose.connect(mongoDB, { useNewUrlParser: true });
var db = mongoose.connection;
db.on('error', console.error.bind(console, 'MongoDB connection error:'));
sample_data = [];

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
  "Reporting Officer": String
})

let Cases = mongoose.model('Cases', caseSchema);
app.get('/', function(req, res) {
  Cases.find({}, function (err, cases) {
    cases.forEach(function(el) {
      
      // we need to fix this, this is such a hack!
      str_arr = String(el).match(/\w+|'[^']+'/g);
      let i = str_arr.indexOf("Location");
      address = str_arr[i+1];
      sample_data.push(address);  
    })
    render(res);
  })
});

function render(res) {
  res.render("test.ejs", {data: sample_data});
}

app.listen(port, () => console.log('Port 3000...'));
