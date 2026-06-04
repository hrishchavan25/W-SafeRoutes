export const ANDHERI_REGIONS = [
  { id: "railway", name: "Andheri Railway Station", lat: 19.1197, lon: 72.8464 },
  { id: "versova", name: "Versova Beach", lat: 19.1351, lon: 72.8146 },
  { id: "dn_nagar", name: "D. N. Nagar", lat: 19.1283, lon: 72.8316 },
  { id: "lokhandwala", name: "Lokhandwala Complex", lat: 19.1433, lon: 72.8244 },
  { id: "oshiwara", name: "Oshiwara", lat: 19.1482, lon: 72.8355 },
  { id: "seven_bungalows", name: "Seven Bungalows", lat: 19.1295, lon: 72.8176 },
  { id: "azad_nagar", name: "Azad Nagar", lat: 19.1279, lon: 72.8371 },
  { id: "juhu_circle", name: "Juhu Circle", lat: 19.1075, lon: 72.8263 },
  { id: "model_town", name: "Model Town", lat: 19.1319, lon: 72.8288 },
  { id: "shastri_nagar", name: "Shastri Nagar", lat: 19.1410, lon: 72.8319 },
];

export function getRegionById(id) {
  return ANDHERI_REGIONS.find((region) => region.id === id) || ANDHERI_REGIONS[0];
}
