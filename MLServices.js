import AsyncStorage from '@react-native-async-storage/async-storage';
import { BACKEND_URL, SOS_BACKEND_URL } from './config';

class MLService {
  static async getApiHost() {
    // We now use config.js as the source of truth as requested
    return BACKEND_URL;
  }



  static async saveApiHost(host) {
    try {
      await AsyncStorage.setItem('API_HOST', host);
      return true;
    } catch (e) {
      return false;
    }
  }

  static async healthCheck() {
    try {
      const host = await this.getApiHost();
      const response = await fetch(`${host.replace(/\/$/, '')}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  static async getZones(options = {}) {
    try {
      const host = await this.getApiHost();
      const city = options.city || "andheri";
      const limit = options.limit || 1000;
      const response = await fetch(`${host.replace(/\/$/, '')}/zones?city=${encodeURIComponent(city)}&limit=${encodeURIComponent(limit)}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      return await response.json();
    } catch (error) {
      console.error('Fetching zones failed:', error);
      throw error;
    }
  }

  static async getSafeRoute(routeParams) {
    try {
      const host = await this.getApiHost();
      const response = await fetch(`${host.replace(/\/$/, '')}/get-safe-route`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(routeParams),
      });
      return await response.json();
    } catch (error) {
      console.error('Safe route fetch failed:', error);
      throw error;
    }
  }

  static async geocode(query, city = 'chicago') {
    try {
      const host = await this.getApiHost();
      const base = host.replace(/\/$/, '');
      const url = `${base}/geocode?q=${encodeURIComponent(query)}&city=${encodeURIComponent(city)}`;
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail;
        const msg =
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg || d).join('; ')
              : `HTTP ${response.status}`;
        throw new Error(msg || 'Geocode failed');
      }
      return data;
    } catch (error) {
      console.error('Geocode failed:', error);
      throw error;
    }
  }

  static async getAStarRoute(routeParams) {
    try {
      const host = await this.getApiHost();
      const response = await fetch(`${host.replace(/\/$/, '')}/api/astar-route`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(routeParams),
      });
      return await response.json();
    } catch (error) {
      console.error('A* route fetch failed:', error);
      throw error;
    }
  }

  static async activateSOS(sosData) {
    try {
      const host = SOS_BACKEND_URL || BACKEND_URL.replace(":8100", ":5000");
      const response = await fetch(`${host.replace(/\/$/, '')}/api/sos/activate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sosData),
      });
      return await response.json();
    } catch (error) {
      console.error('SOS activation failed:', error);
      throw error;
    }
  }
}

export default MLService;
