export async function POST(req: Request) {
  try {
    const payload = await req.json();

    // Prefer NEXT_PUBLIC_API_URL, fallback to HR_API_URL for backward compatibility
    const base =
      process.env.NEXT_PUBLIC_API_URL ||
      process.env.HR_API_URL ||
      "http://localhost:8000";

    const url = base.endsWith("/") ? base.slice(0, -1) : base;
    const target = `${url}/ask/stream`;

    // Create timeout controller
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout
    
    const upstream = await fetch(target, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    }).catch((err) => {
      clearTimeout(timeoutId);
      // Handle network errors (backend offline, CORS, etc.)
      if (err.name === "AbortError") {
        throw new Error("Request timeout: Backend took too long to respond");
      }
      throw new Error(`Backend unreachable at ${target}. Is the server running?`);
    });
    
    clearTimeout(timeoutId);

    if (!upstream.ok) {
      const errText = await upstream.text().catch(() => "Unknown error");
      return new Response(
        JSON.stringify({ 
          error: `Backend error (${upstream.status})`, 
          detail: errText,
          hint: "Check that the backend is running and accessible"
        }),
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
  } catch (e: unknown) {
    const errorMessage = e instanceof Error ? e.message : String(e);
    return new Response(
      JSON.stringify({ error: "Proxy error", detail: errorMessage }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
