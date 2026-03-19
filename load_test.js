import http from 'k6/http';
import { check, fail } from 'k6';

export const options = {
  scenarios: {
    constant_traffic: {
      executor: 'constant-arrival-rate',
      rate: 700,
      timeUnit: '1s',
      duration: '3m',
      preAllocatedVUs: 50,
      maxVUs: 5000,
    },
  },
};

const API_URL = __ENV.API_URL || 'http://localhost:8000';
const PREDICT_BASE_URL = `${API_URL}/predict`;
const LOGIN_URL = `${API_URL}/login`;
const LOGIN = __ENV.LOAD_TEST_LOGIN || 'test_user_1';
const PASSWORD = __ENV.LOAD_TEST_PASSWORD || 'pass123';

function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function getRandomBool() {
  return Math.random() > 0.5;
}

function buildAuthHeaders(accessToken) {
  return {
    'Content-Type': 'application/json',
    Connection: 'keep-alive',
    Cookie: `access_token=${accessToken}`,
  };
}

export function setup() {
  const response = http.post(
    LOGIN_URL,
    JSON.stringify({
      login: LOGIN,
      password: PASSWORD,
    }),
    {
      headers: {
        'Content-Type': 'application/json',
      },
    },
  );

  const loginOk = check(response, {
    'login is 200': (r) => r.status === 200,
    'login returns access token': (r) => Boolean(r.json('access_token')),
  });

  if (!loginOk) {
    fail(
      `Login failed with status ${response.status}. ` +
        'Run insert_data.sh first or override LOAD_TEST_LOGIN/LOAD_TEST_PASSWORD.',
    );
  }

  return {
    accessToken: response.json('access_token'),
  };
}

export default function (data) {
  const headers = buildAuthHeaders(data.accessToken);
  const u = Math.random();

  if (u < 0.45) {
    const payload = JSON.stringify({
      seller_id: getRandomInt(1, 50),
      is_verified_seller: getRandomBool(),
      item_id: getRandomInt(1, 1000),
      name: `Тестовый товар ${getRandomInt(1, 100)}`,
      description: 'Описание товара для проверки ML модели модерации.',
      category: getRandomInt(1, 100),
      images_qty: getRandomInt(0, 10),
    });
    const res = http.post(`${PREDICT_BASE_URL}/`, payload, { headers });
    check(res, { 'predict is 200': (r) => r.status === 200 });
  } else if (u < 0.55) {
    const payload = JSON.stringify({
      seller_id: 'invalid_string',
      is_verified_seller: getRandomBool(),
      item_id: getRandomInt(1, 1000),
      name: 'Сломанный товар',
      category: 1,
      images_qty: -5,
    });
    const res = http.post(`${PREDICT_BASE_URL}/`, payload, { headers });
    check(res, { 'predict validation error is 422': (r) => r.status === 422 });
  } else if (u < 0.8) {
    const itemId = getRandomInt(1, 1000);
    const res = http.get(`${PREDICT_BASE_URL}/simple_predict?item_id=${itemId}`, {
      headers,
    });
    check(res, { 'simple_predict is 200': (r) => r.status === 200 });
  } else if (u < 0.9) {
    const itemId = getRandomInt(900000, 999999);
    const res = http.get(`${PREDICT_BASE_URL}/simple_predict?item_id=${itemId}`, {
      headers,
    });
    check(res, { 'simple_predict error is 404': (r) => r.status === 404 });
  } else {
    const itemId = getRandomInt(1, 1000);
    const res = http.post(
      `${PREDICT_BASE_URL}/async_predict?item_id=${itemId}`,
      null,
      { headers },
    );
    check(res, { 'async_predict is 200': (r) => r.status === 200 });
  }
}
