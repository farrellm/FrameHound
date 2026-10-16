// A short film strip: sprocket edges and a few frames. Used for library rows
// and match rows. Frames are empty until shot thumbnails exist (M2).

export function MiniStrip({ thumbs, count = 5 }: { thumbs: string[]; count?: number }) {
  const cells = Array.from({ length: count }, (_, i) => thumbs[i]);
  return (
    <div className="mini-strip" aria-hidden="true">
      <div className="sprockets" />
      <div className="frames">
        {cells.map((t, i) => (
          <div key={i} className="frame" style={t ? { backgroundImage: `url(${t})` } : undefined} />
        ))}
      </div>
      <div className="sprockets" />
    </div>
  );
}
