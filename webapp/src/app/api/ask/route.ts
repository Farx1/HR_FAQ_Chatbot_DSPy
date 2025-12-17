export async function POST(req: Request) {
  try {
    const payload = await req.json();

    const base =
      process.env.HR_API_URL ||
      process.env.NEXT_PUBLIC_API_URL ||
      "http://localhost:8000";

    const url = base.endsWith("/") ? base.slice(0, -1) : base;
    const target = `${url}/ask/stream`;

    const upstream = await fetch(target, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!upstream.ok) {
      const errText = await upstream.text().catch(() => "Unknown error");
      return new Response(
        JSON.stringify({ error: errText }),
        { status: upstream.status, headers: { "Content-Type": "application/json" } }
      );
    }

    // Always stream the response
    return new Response(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  } catch (e: any) {
    return new Response(
      JSON.stringify({ error: "Proxy error", detail: String(e?.message ?? e) }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
