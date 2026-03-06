import http from 'k6/http';
import { check } from 'k6';

export const options = {
  scenarios: {
    constant_traffic: {
      executor: 'constant-arrival-rate',
      rate: 500,
      timeUnit: '1s',
      duration: '10m',
      preAllocatedVUs: 50,
      maxVUs: 5000,
    },
  },
};

function getRandomInt(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }
function getRandomBool() { return Math.random() > 0.5; }

const BASE_URL = 'http://localhost:8000/predict';
const HEADERS = { 'Content-Type': 'application/json', 'Connection': 'keep-alive' };

export default function () {
  const u = Math.random();

  if (u < 0.40) {
    const payload = JSON.stringify({
      seller_id: getRandomInt(1, 50),
      is_verified_seller: getRandomBool(),
      item_id: getRandomInt(1, 1000),
      name: "Тестовый товар " + getRandomInt(1, 100),
      description: "Описание товара для проверки ML модели модерации.",
      category: getRandomInt(1, 100),
      images_qty: getRandomInt(0, 10)
    });
    const res = http.post(`${BASE_URL}/`, payload, { headers: HEADERS });
    check(res, { 'predict is 200': (r) => r.status === 200 });

  } else if (u < 0.55) {
    const payload = JSON.stringify({
      seller_id: "invalid_string",
      is_verified_seller: getRandomBool(),
      item_id: getRandomInt(1, 1000),
      name: "Сломанный товар",
      category: 1,
      images_qty: -5
    });
    const res = http.post(`${BASE_URL}/`, payload, { headers: HEADERS });
    check(res, { 'predict error is 422': (r) => r.status === 422 });

  } else if (u < 0.75) {
    const itemId = getRandomInt(1, 1000);
    const res = http.get(`${BASE_URL}/simple_predict?item_id=${itemId}`, { headers: HEADERS });
    check(res, { 'simple_predict is 200': (r) => r.status === 200 });

  } else if (u < 0.85) {
    const itemId = getRandomInt(900000, 999999);
    const res = http.get(`${BASE_URL}/simple_predict?item_id=${itemId}`, { headers: HEADERS });
    check(res, { 'simple_predict error is 404': (r) => r.status === 404 });

  } else {
    const itemId = getRandomInt(1, 1000);
    const res = http.post(`${BASE_URL}/async_predict?item_id=${itemId}`, null, { headers: HEADERS });
    check(res, { 'async_predict is 200': (r) => r.status === 200 });
  }
}