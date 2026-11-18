import { Link } from "@tanstack/react-router";
import { ARCHETYPES, NODES } from "@/lib/cams/model";

export function PrimerView() {
  return (
    <article className="mx-auto max-w-2xl px-4 py-8 md:py-12">
      <p className="font-mono text-xs tracking-widest text-brass">HOW TO READ THE INSTRUMENT</p>
      <h1 className="mt-3 font-display text-5xl leading-none text-balance">Eight nodes, one value.</h1>
      <p className="mt-5 text-pretty text-lg text-mute">
        A society here is not a mood and not a morality. It is eight institutions, each scored for coherence,
        capacity, stress and abstraction. The atlas shows the JUNO panel with the names on. Blind Prophet covers the
        names and asks you to call the break.
      </p>

      <p className="mt-10 font-display text-3xl italic text-balance">V = C + K − S + ½A</p>
      <p className="mt-3 text-pretty text-mute">
        C is coherence, K capacity, S stress, A abstraction. V is that node’s value. V̄ is the mean of the eight. Vmin
        is the worst of them. Under 4, the year is strained. Through zero, it is severe.
      </p>

      <h2 className="mt-10 font-display text-3xl">The nodes</h2>
      <dl className="mt-4 divide-y divide-line">
        {NODES.map((node) => (
          <div key={node.id} className="grid grid-cols-[8rem_1fr] gap-3 py-3">
            <dt className="font-mono text-sm text-brass">{node.id}</dt>
            <dd className="text-mute">
              {node.gloss}
              <span className="text-faint"> · {node.family}</span>
            </dd>
          </div>
        ))}
      </dl>

      <h2 className="mt-10 font-display text-3xl">Bonds</h2>
      <p className="mt-3 text-pretty text-mute">
        A bond is strong when both nodes are coherent and abstract, and it weakens as either one is stressed.
      </p>
      <p className="mt-3 font-mono text-sm text-bone">h = (0.6C + 0.4A) / 10</p>
      <p className="font-mono text-sm text-bone">Bᵢⱼ = √(hᵢ hⱼ) · 2^(−(Sᵢ+Sⱼ)/10)</p>
      <ul className="mt-4 space-y-2 text-mute">
        <li>
          <span className="text-bone">B̄</span> is the mean of the 28 unique bonds.
        </li>
        <li>
          <span className="text-bone">λ₂</span> is the algebraic connectivity of that graph. Low, and the society can
          come apart.
        </li>
        <li>
          <span className="text-bone">η</span> is the thinner of Lore’s and Archive’s bond strength, divided by how
          stressed Hands are. That is the memory floor.
        </li>
        <li>
          <span className="text-bone">ω</span> is how unevenly the eight nodes moved on this step.
        </li>
      </ul>

      <h2 className="mt-10 font-display text-3xl">Signatures</h2>
      <ul className="mt-4 space-y-3 text-mute">
        <li>
          <span className="text-bone">Executive decoupling.</span> Helm’s value is under 6 and its bond to Archive is
          thin.
        </li>
        <li>
          <span className="text-bone">Praetorian condition.</span> Shield outvalues both Helm and the Stewards. A high
          V̄ in that state is not health.
        </li>
        <li>
          <span className="text-bone">Mythic-material decoupling.</span> Lore is carrying the system while Hands have
          failed.
        </li>
        <li>
          <span className="text-bone">Thermodynamic freeze.</span> Mean value is at or below zero and the memory floor
          has collapsed.
        </li>
      </ul>

      <h2 className="mt-10 font-display text-3xl">Four shapes</h2>
      <dl className="mt-4 space-y-4">
        {ARCHETYPES.map((item) => (
          <div key={item.id}>
            <dt className="text-bone">{item.name}</dt>
            <dd className="text-mute">{item.blurb}</dd>
          </div>
        ))}
      </dl>

      <h2 className="mt-10 font-display text-3xl">Calling a crisis</h2>
      <p className="mt-3 text-pretty text-mute">
        In Blind Prophet the years are an index, not a date. Two calls. A call one to five samples before an onset is a
        hit, worth 60 plus 25 for each year of lead, capped at 160. Inside five samples after, it is late, worth 40.
        Anything else is a false call, minus 50. Guessing the society is 100. Guessing the shape is another 100. If the
        window never onset and you never called, the discipline bonus is 120.
      </p>
      <p className="mt-6 flex flex-wrap gap-4 text-sm">
        <Link to="/" className="text-brass">
          Open the atlas
        </Link>
        <Link to="/prophet" className="text-brass">
          Play Blind Prophet
        </Link>
        <a href="https://neuralnations.org" className="text-mute hover:text-bone">
          neuralnations.org
        </a>
      </p>
    </article>
  );
}
