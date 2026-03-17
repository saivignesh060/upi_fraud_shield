import { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { Network } from 'lucide-react';

interface Props {
  transactions: any[];
}

export default function NetworkGraph({ transactions }: Props) {
  const d3Container = useRef<SVGSVGElement | null>(null);
  const simulationRef = useRef<d3.Simulation<any, any> | null>(null);
  const nodesMapRef = useRef<Map<string, any>>(new Map());

  // Parse transactions into nodes & edges (sliding window limits)
  const graphData = useMemo(() => {
      const windowTxns = transactions.slice(0, 500);
      
      const userNodes = new Map();
      const mchNodes = new Map();
      const links: any[] = [];
      
      for (const t of windowTxns) {
          if (userNodes.size >= 80 && !userNodes.has(t.user_id)) continue;
          if (mchNodes.size >= 40 && !mchNodes.has(t.merchant_category)) continue;

          if (t.should_display_alert || Math.random() < 0.2) {
              if (!userNodes.has(t.user_id)) {
                  userNodes.set(t.user_id, { id: t.user_id, type: 'user', risk: 0, count: 0 });
              }
              if (!mchNodes.has(t.merchant_category)) {
                  mchNodes.set(t.merchant_category, { id: t.merchant_category, type: 'merchant', count: 0 });
              }
              
              const u = userNodes.get(t.user_id);
              u.count++;
              if (t.should_display_alert) u.risk += 1;
              
              const m = mchNodes.get(t.merchant_category);
              m.count++;

              links.push({
                  source: t.user_id,
                  target: t.merchant_category,
                  isFraud: t.should_display_alert,
                  value: t.amount,
                  id: `${t.user_id}-${t.merchant_category}-${t.transaction_id}`
              });
          }
      }

      const nodes = [...Array.from(userNodes.values()), ...Array.from(mchNodes.values())];
      return { nodes, links };
  }, [transactions]);

  useEffect(() => {
    if (!d3Container.current || graphData.nodes.length === 0) return;

    const width = d3Container.current.clientWidth;
    const height = d3Container.current.clientHeight;
    const svg = d3.select(d3Container.current);

    // Initialize SVG structure and simulation once
    if (!simulationRef.current) {
        const defs = svg.append("defs");
        const filter = defs.append("filter").attr("id", "glow");
        filter.append("feGaussianBlur").attr("stdDeviation", "2.5").attr("result", "coloredBlur");
        const feMerge = filter.append("feMerge");
        feMerge.append("feMergeNode").attr("in", "coloredBlur");
        feMerge.append("feMergeNode").attr("in", "SourceGraphic");

        svg.append("g").attr("class", "links");
        svg.append("g").attr("class", "nodes");

        simulationRef.current = d3.forceSimulation()
            .force("link", d3.forceLink().id((d: any) => d.id).distance(60))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(20))
            .alphaDecay(0.05)
            .velocityDecay(0.4);
    }

    const validLinksData = graphData.links.filter(l => 
        graphData.nodes.some(n => n.id === l.source) && 
        graphData.nodes.some(n => n.id === l.target)
    );

    const now = Date.now();
    const oldMap = nodesMapRef.current;
    const newMap = new Map<string, any>();
    
    let hasNewNodes = false;
    
    const validNodesData = graphData.nodes.map(d => {
        const existing = oldMap.get(d.id);
        if (existing) {
            existing.risk = d.risk;
            existing.count = d.count;
            if (now - existing.createdAt > 3000) {
                existing.fx = existing.x;
                existing.fy = existing.y;
            }
            newMap.set(d.id, existing);
            return existing;
        } else {
            hasNewNodes = true;
            const newNode = { ...d, createdAt: now };
            newMap.set(d.id, newNode);
            return newNode;
        }
    });

    nodesMapRef.current = newMap;
    const validLinks = validLinksData.map(d => ({ ...d }));

    const simulation = simulationRef.current;

    simulation.nodes(validNodesData);
    const linkForce = simulation.force("link") as d3.ForceLink<any, any>;
    linkForce.links(validLinks);

    if (hasNewNodes) {
        simulation.alpha(0.3).restart();
    }

    const linkGroup = svg.select("g.links");
    const nodeGroup = svg.select("g.nodes");

    const link = linkGroup.selectAll("line")
        .data(validLinks, (d: any) => d.target.id ? `${d.source.id}-${d.target.id}` : `${d.source}-${d.target}`)
        .join("line")
        .attr("stroke", (d: any) => d.isFraud ? "#da3633" : "#30363D")
        .attr("stroke-opacity", (d: any) => d.isFraud ? 0.8 : 0.4)
        .attr("stroke-width", (d: any) => d.isFraud ? 2 : 1);

    const node = nodeGroup.selectAll("circle")
        .data(validNodesData, (d: any) => d.id)
        .join("circle")
        .attr("r", (d: any) => d.type === 'merchant' ? 12 : 6)
        .attr("fill", (d: any) => d.type === 'merchant' ? "#d29922" : (d.risk > 0 ? "#da3633" : "#58a6ff"))
        .style("filter", (d: any) => d.risk > 0 ? "url(#glow)" : "none")
        .call(d3.drag()
            .on("start", (event: any, d: any) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
                d.fixedByDrag = true;
            })
            .on("drag", (event: any, d: any) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on("end", (event: any, d: any) => {
                if (!event.active) simulation.alphaTarget(0);
                if (!d.fixedByDrag && (now - d.createdAt <= 3000)) {
                    d.fx = null;
                    d.fy = null;
                }
            }) as any);

    node.selectAll("title").remove(); 
    node.append("title")
        .text((d: any) => `${d.type === 'merchant' ? 'C' : 'U'} ${d.id}\nTx Count: ${d.count}`);

    simulation.on("tick", () => {
        link
            .attr("x1", (d: any) => d.source.x)
            .attr("y1", (d: any) => d.source.y)
            .attr("x2", (d: any) => d.target.x)
            .attr("y2", (d: any) => d.target.y);

        node
            .attr("cx", (d: any) => Math.max(12, Math.min(width - 12, d.x)))
            .attr("cy", (d: any) => Math.max(12, Math.min(height - 12, d.y)));
    });

  }, [graphData]);

  return (
    <div className="flex flex-col h-full w-full">
      <div className="p-4 border-b border-[#30363D] bg-[#161B22] flex items-center justify-between shrink-0">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Network size={18} className="text-indigo-500" />
            Entity Fraud Network Graph
        </h2>
        <div className="flex gap-4 text-xs font-semibold text-gray-400">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500"></span> User</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-yellow-600"></span> Merchant</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_red]"></span> Risky Cluster</span>
        </div>
      </div>
      <div className="flex-1 w-full relative z-0 overflow-hidden bg-[#0D1117]/50">
          <svg className="w-full h-full" ref={d3Container} />
      </div>
    </div>
  );
}
