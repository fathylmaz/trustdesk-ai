import React from 'react';

export default function Icon({ name, size=20, className='', ...props }) {
  const paths = {
    spark: <><path d="m12 2 2.2 6.8L21 11l-6.8 2.2L12 20l-2.2-6.8L3 11l6.8-2.2L12 2Z" /><path d="m19 18 .75 2.25L22 21l-2.25.75L19 24l-.75-2.25L16 21l2.25-.75L19 18Z" /></>,
    layers: <><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M3 9h18M9 9v11"/></>,
    bolt: <path d="m13 2-9 12h7l-1 8 10-13h-7l0-7Z"/>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    x: <path d="M6 6 18 18M18 6 6 18"/>,
    refresh: <><path d="M20 11a8 8 0 0 0-14-5L3 9"/><path d="M3 3v6h6"/><path d="M4 13a8 8 0 0 0 14 5l3-3"/><path d="M21 21v-6h-6"/></>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    globe: <><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c-5 5-5 13 0 18M12 3c5 5 5 13 0 18"/></>,
    shield: <><path d="M12 2 4 5v6c0 6 3.5 9 8 11 4.5-2 8-5 8-11V5l-8-3Z"/><path d="m9 12 2 2 4-4"/></>,
    terminal: <><rect x="2" y="3" width="20" height="18" rx="3"/><path d="m6 8 3 3-3 3M12 15h5"/></>,
    alert: <><path d="M12 3 2 21h20L12 3Z"/><path d="M12 9v5m0 3h.01"/></>,
    copy: <><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></>,
  };
  return <svg aria-hidden="true" width={size} height={size} className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...props}>{paths[name] || paths.spark}</svg>;
}
