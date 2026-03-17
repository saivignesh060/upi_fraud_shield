import GeoMap from './GeoMap';
import NetworkGraph from './NetworkGraph';

interface Props {
  transactions: any[];
  alerts: any[];
}

export default function ThreatIntelligence({ transactions, alerts }: Props) {
  return (
    <div className="flex-1 min-h-0 flex gap-6 px-6 pb-6">
      <div className="bg-[#161B22] border border-[#30363D] rounded-xl overflow-hidden shadow-lg flex-1 min-h-0 min-w-0 flex flex-col">
        <GeoMap transactions={transactions} alerts={alerts} />
      </div>
      <div className="bg-[#161B22] border border-[#30363D] rounded-xl overflow-hidden shadow-lg flex-1 min-h-0 min-w-0 flex flex-col">
        <NetworkGraph transactions={transactions} />
      </div>
    </div>
  );
}
