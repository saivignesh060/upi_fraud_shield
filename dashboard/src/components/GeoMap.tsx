import { useMemo, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Map } from 'lucide-react';

const CITY_COORDS: Record<string, [number, number]> = {
  'Hyderabad': [17.3850, 78.4867], 'Kochi': [9.9312, 76.2673],
  'Kolkata': [22.5726, 88.3639], 'Pune': [18.5204, 73.8567],
  'Delhi': [28.7041, 77.1025], 'Chennai': [13.0827, 80.2707],
  'Mumbai': [19.0760, 72.8777], 'Ahmedabad': [23.0225, 72.5714],
  'Surat': [21.1702, 72.8311], 'Gurugram': [28.4595, 77.0266],
  'Noida': [28.5355, 77.3910], 'Bengaluru': [12.9716, 77.5946],
  'Jaipur': [26.9124, 75.7873], 'Indore': [22.7196, 75.8577],
  'Nagpur': [21.1458, 79.0882], 'Ludhiana': [30.9010, 75.8523],
  'Bhopal': [23.2599, 77.4126], 'Chandigarh': [30.7333, 76.7794],
  'Vizag': [17.6868, 83.2185], 'Patna': [25.5941, 85.1376],
  'Lucknow': [26.8467, 80.9462]
};

interface Props {
  transactions: any[];
  alerts: any[];
}

export default function GeoMap({ transactions, alerts }: Props) {
  // Fix Leaflet container size bug by triggering resize.
  useEffect(() => {
    const handleResize = () => {
      window.dispatchEvent(new Event('resize'));
    };
    setTimeout(handleResize, 100);
  }, []);

  const { txns, arcs } = useMemo(() => {
    // Only plot last 200
    const txns = transactions.slice(0, 200).filter(t => CITY_COORDS[t.location_city]);
    
    // Calculate last 20 location jumps
    const jumpAlerts = alerts.filter(a => a.fraud_type_predicted === 'location_jump').slice(0, 20);
    const arcs: { id: string, from: [number, number], to: [number, number] }[] = [];
    
    for (const alert of jumpAlerts) {
      if (!CITY_COORDS[alert.location_city]) continue;
      // find previous txn for same user
      const userTxns = transactions.filter(t => t.user_id === alert.user_id && CITY_COORDS[t.location_city]);
      const idx = userTxns.findIndex(t => t.transaction_id === alert.transaction_id);
      if (idx !== -1 && idx + 1 < userTxns.length) {
        const prevTxn = userTxns[idx + 1]; // array is reversed (newest first)
        if (prevTxn.location_city !== alert.location_city) {
            arcs.push({
                id: alert.transaction_id,
                from: CITY_COORDS[prevTxn.location_city],
                to: CITY_COORDS[alert.location_city]
            });
        }
      }
    }
    
    return { txns, arcs };
  }, [transactions, alerts]);

  return (
    <div className="flex flex-col h-full w-full">
      <div className="p-4 border-b border-[#30363D] bg-[#161B22] flex items-center justify-between shrink-0">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Map size={18} className="text-emerald-500" />
            Live Geo-Spatial Telemetry
        </h2>
        <div className="flex gap-4 text-xs font-semibold text-gray-400">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> Legit</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500"></span> Flagged</span>
            <span className="flex items-center gap-1"><span className="w-4 h-0.5 bg-red-400"></span> Jump Arc</span>
        </div>
      </div>
      <div className="flex-1 w-full relative z-0">
        <MapContainer 
            center={[21.0, 78.0]} 
            zoom={4.5} 
            className="w-full h-full"
            style={{ backgroundColor: '#0D1117' }}
            attributionControl={false}
        >
          <TileLayer
            url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png"
          />
          
          {arcs.map(arc => (
             <Polyline 
                key={arc.id}
                positions={[arc.from, arc.to]} 
                pathOptions={{ color: '#da3633', weight: 3, dashArray: '5, 5', className: 'animate-pulse opacity-70' }} 
             />
          ))}

          {txns.map(t => (
            <CircleMarker
              key={t.transaction_id}
              center={CITY_COORDS[t.location_city]}
              radius={t.should_display_alert ? 8 : 4}
              pathOptions={{ 
                  color: t.should_display_alert ? '#da3633' : '#238636', 
                  fillColor: t.should_display_alert ? '#da3633' : '#238636',
                  fillOpacity: 0.8,
                  weight: 1
              }}
            >
              <Popup className="bg-[#161B22] border-[#30363D] text-gray-200">
                <div className="text-sm">
                    <strong>ID:</strong> {t.transaction_id.substring(0,8)}<br/>
                    <strong>City:</strong> {t.location_city}<br/>
                    <strong>Amount:</strong> ₹{t.amount}<br/>
                    <strong>Score:</strong> {(t.risk_score * 100).toFixed(1)}%<br/>
                    {t.should_display_alert && (
                        <strong className="text-red-400 mt-1 block drop-shadow-md">{t.fraud_type_predicted}</strong>
                    )}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  );
}
