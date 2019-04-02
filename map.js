const express = require('express');
const app = express();
const port = 3000;
app.set('view engine', 'ejs');
var mongoose = require('mongoose');

var mongoDB = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data';

mongoose.connect(mongoDB, { useNewUrlParser: true });
var db = mongoose.connection;
db.on('error', console.error.bind(console, 'MongoDB connection error:'));
sample_data = ["hi", "hey", "hello"];

let caseSchema = new mongoose.Schema({
  "_id": String,
  "Arrest Date": String,
  "Year": Number,
  "Month": Number,
  "Last Name": String,
  "First Name": String,
  "Gender": String,
  "Race": String,
  "Ethnicity": String,
  "Year of Birth": Number,
  "Age": Number,
  "From Address": String,
  "From City": String,
  "From State": String,
  "Statute Type": String,
  "Statute Code": String,
  "Statute Desc": String,
  "Counts": Number,
  "Case Number": String,
  "Arresting Officers": String
})

let Cases = mongoose.model('Cases', caseSchema);
app.get('/', function(req, res) {
  // Cases.find({}, function (err, cases) {
  //   cases.forEach(function(el) {
  //     console.log(el);
  //   })
  // })
  res.render("test.ejs", {data: sample_data});
});

app.listen(port, () => console.log('Port 3000...'));