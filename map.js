import { Stitch } from 'mongodb-stitch-browser-sdk';
import {
  HttpServiceClient,
  HttpRequest,
  HttpMethod
} from 'mongodb-stitch-browser-services-http';
// 1. Instantiate an HTTP Service Client
const app = Stitch.defaultAppClient;
const http = app.getServiceClient(HttpServiceClient.factory, "myHttp");

// 2. Build a new HttpRequest
const request = new HttpRequest.Builder()
  .withMethod(HttpMethod.GET)
  .withUrl("https://www.example.com/users")
  .build()

// 3. Execute the built request
http.execute(request)
  .then(console.log)
  .catch(console.error)
