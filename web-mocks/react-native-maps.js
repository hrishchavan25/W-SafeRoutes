import React from 'react';
import { View } from 'react-native';

export const PROVIDER_GOOGLE = 'google';

export const Marker = ({ children, ...props }) => {
  return React.createElement(View, props, children || null);
};

export const Polyline = ({ children, ...props }) => {
  return React.createElement(View, props, children || null);
};

const MapView = ({ children, style, ...props }) => {
  const fallbackStyle = { backgroundColor: '#eef', minHeight: 200 };
  const mergedStyle = Array.isArray(style) ? [fallbackStyle, ...style] : { ...fallbackStyle, ...(style || {}) };
  return React.createElement(View, { style: mergedStyle, ...props }, children || null);
};

export default MapView;
