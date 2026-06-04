import time
import urllib.request
import json


def wait_health(url='http://127.0.0.1:5000/health', retries=20, delay=0.5):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                print('HEALTH_RESPONSE:', r.read().decode())
                return True
        except Exception as e:
            time.sleep(delay)
    print('HEALTH_CHECK_FAILED')
    return False


def test_predict(url='http://127.0.0.1:5000/predict'):
    data = {'latitude': 19.1240, 'longitude': 72.8254}
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            print('PREDICT_RESPONSE:', r.read().decode())
    except Exception as e:
        print('PREDICT_ERROR:', e)


if __name__ == '__main__':
    ok = wait_health()
    if ok:
        test_predict()
