"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ActionIcon,
  AppShell,
  Badge,
  Box,
  Button,
  Card,
  Divider,
  Group,
  Paper,
  Progress,
  ScrollArea,
  SegmentedControl,
  Stack,
  Tabs,
  Text,
  Textarea,
  ThemeIcon,
  Title,
  Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import {
  IconBrandGithub,
  IconBuilding,
  IconClock,
  IconFileText,
  IconMessage2,
  IconSearch,
  IconShieldCheck,
  IconSparkles,
  IconSubtask,
  IconUser,
  IconWand,
} from "@tabler/icons-react";

type Role = "user" | "assistant";

type Source = { title?: string; url?: string; snippet?: string };

type ApiAnswer = {
  answer?: string;
  response?: string;
  output?: string;
  ood_reject?: boolean;
  confidence?: number; // 0..1
  latency_ms?: number;
  sources?: Source[];
  [k: string]: any;
};

type Message = { id: string; role: Role; content: string; ts: number };

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function pickAnswer(r: ApiAnswer) {
  return r.answer ?? r.response ?? r.output ?? "No answer field returned by API.";
}

async function postJSON<T>(url: string, body: any, timeoutMs = 15000): Promise<T> {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

// Stream response from SSE endpoint
async function streamResponse(
  url: string,
  body: any,
  onChunk: (text: string) => void,
  onDone: (meta: any) => void,
  onError: (err: Error) => void
) {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No reader");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === "chunk") {
              onChunk(data.content);
            } else if (data.type === "done") {
              onDone(data);
            }
          } catch {}
        }
      }
    }
  } catch (err) {
    onError(err as Error);
  }
}

// @ts-expect-error - Mantine Card polymorphic types conflict with framer-motion
const MotionCard = motion.create(Card);

export default function Page() {
  const reduceMotion = useReducedMotion();

  const [mode, setMode] = useState<"policy" | "benefits" | "payroll">("policy");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const [messages, setMessages] = useState<Message[]>(() => [
    {
      id: uid(),
      role: "assistant",
      ts: Date.now(),
      content:
        "Welcome 👋\nI’m your HR FAQ Copilot.\nAsk about leave, remote work, benefits, payroll, onboarding, and training.\n\nTry: “How many vacation days do I get per year?”",
    },
  ]);

  const [meta, setMeta] = useState<{
    ood?: boolean;
    latency?: number;
    confidence?: number;
    sources?: Source[];
  }>({});

  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: reduceMotion ? "auto" : "smooth" });
  }, [messages, reduceMotion]);

  const topics = useMemo(
    () => [
      { label: "Leave & PTO", icon: IconSubtask, hint: "vacation, sick leave, holidays" },
      { label: "Remote work", icon: IconBuilding, hint: "WFH, equipment, travel policy" },
      { label: "Benefits", icon: IconSparkles, hint: "healthcare, perks, reimbursements" },
      { label: "Payroll", icon: IconFileText, hint: "payslip, bank info, delays" },
      { label: "Training", icon: IconWand, hint: "budget, approvals, onboarding" },
    ],
    []
  );

  const suggestions = useMemo(() => {
    if (mode === "benefits")
      return [
        "What benefits are available for employees?",
        "How does the reimbursement policy work?",
        "Do we have a wellness or sport allowance?",
      ];
    if (mode === "payroll")
      return [
        "How do I update my bank details for payroll?",
        "When are payslips available each month?",
        "Who do I contact if my salary is delayed?",
      ];
    return [
      "How many vacation days do I get per year?",
      "What is the remote work policy?",
      "How do I request sick leave?",
    ];
  }, [mode]);

  async function send(text?: string) {
    const q = (text ?? query).trim();
    if (!q || loading) return;

    setLoading(true);
    setQuery("");

    const userMsg: Message = { id: uid(), role: "user", ts: Date.now(), content: q };
    setMessages((m) => [...m, userMsg]);

    // Create placeholder for streaming response
    const assistantMsgId = uid();
    const assistantMsg: Message = {
      id: assistantMsgId,
      role: "assistant",
      ts: Date.now(),
      content: "",
    };
    setMessages((m) => [...m, assistantMsg]);

    const history = messages.slice(-10).map((m) => ({ role: m.role, content: m.content }));

    await streamResponse(
      "/api/ask",
      { question: q, mode, history },
      // onChunk - update message content
      (chunk) => {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, content: msg.content + chunk }
              : msg
          )
        );
      },
      // onDone - set metadata
      (meta) => {
        const ood = Boolean(meta.ood_reject);
        setMeta({
          ood,
          latency: typeof meta.latency_ms === "number" ? Math.round(meta.latency_ms) : undefined,
          confidence: typeof meta.confidence === "number" ? meta.confidence : undefined,
          sources: Array.isArray(meta.sources) ? meta.sources : undefined,
        });

        // Prepend OOD warning if needed
        if (ood) {
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantMsgId
                ? { ...msg, content: `⚠️ Out-of-domain detected.\n\n${msg.content}` }
                : msg
            )
          );
        }

        setLoading(false);
      },
      // onError
      (err) => {
        notifications.show({
          title: "API unreachable",
          message: String(err?.message ?? err),
          color: "orange",
        });

        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  content:
                    "I can't reach the API right now.\n\nCheck `NEXT_PUBLIC_API_URL` in `.env.local` (default: http://localhost:8000) and confirm your backend is running.\nStart the backend with: uvicorn backend.server:app --reload --port 8000",
                }
              : msg
          )
        );

        setLoading(false);
      }
    );
  }

  return (
    <Box
      style={{
        minHeight: "100vh",
        background:
          "radial-gradient(1200px 600px at 10% 0%, rgba(99,102,241,0.18), transparent 60%), radial-gradient(900px 500px at 90% 10%, rgba(236,72,153,0.14), transparent 55%), radial-gradient(1100px 700px at 50% 110%, rgba(34,197,94,0.10), transparent 55%), #f6f7fb",
      }}
    >
      <AppShell
        padding="md"
        header={{ height: 72 }}
        navbar={{ width: 320, breakpoint: "md" }}
        aside={{ width: 360, breakpoint: "lg" }}
      >
        {/* Header */}
        <AppShell.Header style={{ background: "rgba(255,255,255,0.75)", backdropFilter: "blur(16px)" }}>
          <Group h="100%" px="md" justify="space-between">
            <Group gap="sm">
              <ThemeIcon radius="md" size={40} variant="filled" color="dark">
                <IconShieldCheck size={20} />
              </ThemeIcon>
              <Box>
                <Title order={4} fw={800} lh={1.1} c="rgba(62, 60, 60, 1)">
                  HR FAQ Copilot
                </Title>
                <Text size="sm" c="dimmed">
                  DSPy-enhanced • guardrails • citations-ready
                </Text>
              </Box>
            </Group>

            <Group gap="sm">
              <SegmentedControl
                value={mode}
                onChange={(v) => setMode(v as any)}
                data={[
                  { label: "Policy", value: "policy" },
                  { label: "Benefits", value: "benefits" },
                  { label: "Payroll", value: "payroll" },
                ]}
              />
              <Tooltip label="Repository" withArrow>
                <ActionIcon
                  component="a"
                  href="https://github.com/Farx1/HR_FAQ_Chatbot_DSPy"
                  target="_blank"
                  rel="noreferrer"
                  variant="light"
                  radius="md"
                  size="lg"
                >
                  <IconBrandGithub size={20} />
                </ActionIcon>
              </Tooltip>
            </Group>
          </Group>
        </AppShell.Header>

        {/* Sidebar */}
        <AppShell.Navbar p="md" style={{ background: "rgba(255,255,255,0.78)", backdropFilter: "blur(14px)" }}>
          <Stack gap="md">
            <Paper radius="lg" p="md" withBorder>
              <Group justify="space-between" mb={6}>
                <Group gap="xs" c="rgba(62, 60, 60, 1)">
                  <IconMessage2 size={18} />
                  <Text fw={800}>Topics</Text>
                </Group>
                <Badge variant="light">HR</Badge>
              </Group>

              <Stack gap="sm">
                {topics.map((t) => (
                  <Card
                    key={t.label}
                    radius="md"
                    withBorder
                    padding="sm"
                    style={{ cursor: "pointer" }}
                    onClick={() => setQuery(`Tell me about ${t.label.toLowerCase()}.`)}
                  >
                    <Group gap="sm" wrap="nowrap">
                      <ThemeIcon radius="md" variant="light">
                        <t.icon size={18} />
                      </ThemeIcon>
                      <Box style={{ minWidth: 0 }}>
                        <Text fw={700} truncate>
                          {t.label}
                        </Text>
                        <Text size="sm" c="dimmed" truncate>
                          {t.hint}
                        </Text>
                      </Box>
                    </Group>
                  </Card>
                ))}
              </Stack>
            </Paper>

            <Paper radius="lg" p="md" withBorder>
              <Group gap="xs" mb={6}>
                <IconSearch size={18} />
                <Text fw={800}>Quick prompts</Text>
              </Group>

              <Stack gap="xs">
                {suggestions.map((s) => (
                  <Button
                    key={s}
                    variant="light"
                    radius="md"
                    justify="space-between"
                    onClick={() => send(s)}
                    disabled={loading}
                  >
                    <Text size="sm" truncate>
                      {s}
                    </Text>
                  </Button>
                ))}
              </Stack>

              <Divider my="md" />
              <Text size="sm" c="dimmed">
                Tip: <b>Enter</b> sends • <b>Shift+Enter</b> new line
              </Text>
            </Paper>
          </Stack>
        </AppShell.Navbar>

        {/* Main Chat */}
        <AppShell.Main>
          <Card radius="xl" withBorder shadow="sm" style={{ background: "rgba(255,255,255,0.82)", backdropFilter: "blur(14px)" }}>
            <Group justify="space-between" p="md">
              <Box>
                <Title order={5} fw={850}>
                  Conversation
                </Title>
                <Text size="sm" c="dimmed">
                  Ask questions. We’ll flag out-of-domain queries to reduce hallucinations.
                </Text>
              </Box>
              <Group gap="xs">
                <Badge variant="light" color={meta.ood ? "orange" : "green"}>
                  {meta.ood ? "OOD flagged" : "In-domain"}
                </Badge>
                {typeof meta.latency === "number" && (
                  <Badge variant="light" color="blue" leftSection={<IconClock size={14} />}>
                    {meta.latency}ms
                  </Badge>
                )}
              </Group>
            </Group>

            <Divider />

            <Box ref={scrollRef} style={{ height: "calc(100vh - 260px)" }}>
              <ScrollArea h="100%" p="md">
                <Stack gap="md">
                  <AnimatePresence initial={false}>
                    {messages.map((m) => (
                      <MessageBubble key={m.id} msg={m} reduceMotion={reduceMotion ?? false} />
                    ))}
                  </AnimatePresence>

                  {loading && (
                    <Group align="flex-start" gap="sm">
                      <ThemeIcon radius="xl" variant="filled" color="dark" size={34}>
                        <IconShieldCheck size={16} />
                      </ThemeIcon>
                      <Card radius="lg" withBorder p="sm" style={{ background: "#f4f6ff" }}>
                        <Text size="sm" c="dimmed">
                          Thinking…
                        </Text>
                        <Progress mt={8} size="sm" value={60} animated />
                      </Card>
                    </Group>
                  )}
                </Stack>
              </ScrollArea>
            </Box>

            <Divider />

            {/* Composer */}
            <Box p="md">
              <Group gap="sm" align="flex-end">
                <Textarea
                  value={query}
                  onChange={(e) => setQuery(e.currentTarget.value)}
                  placeholder="Ask a question about HR policy…"
                  autosize
                  minRows={2}
                  maxRows={5}
                  style={{ flex: 1 }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      send();
                    }
                  }}
                />
                <Button loading={loading} onClick={() => send()} radius="md" leftSection={<IconUser size={16} />}>
                  Send
                </Button>
              </Group>

              <Text size="xs" c="dimmed" mt="xs">
                Optional: backend can return <code>sources: [{`{title,url,snippet}`}]</code> for citations.
              </Text>
            </Box>
          </Card>
        </AppShell.Main>

        {/* Aside (Quality & Sources) */}
        <AppShell.Aside p="md" style={{ background: "rgba(255,255,255,0.78)", backdropFilter: "blur(14px)" }}>
          <Stack gap="md">
            <Paper radius="lg" p="md" withBorder>
              <Group justify="space-between" mb={6}>
                <Text fw={850} c="rgba(62, 60, 60, 1)">Quality panel</Text>
                <Badge variant="light" leftSection={<IconShieldCheck size={14} />}>
                  guardrails
                </Badge>
              </Group>

              <Stack gap="sm">
                <KPI title="OOD reject" value="enabled" color={meta.ood ? "orange" : "green"} />
                <KPI
                  title="Confidence"
                  value={
                    typeof meta.confidence === "number"
                      ? `${Math.round(meta.confidence * 100)}%`
                      : "—"
                  }
                />
                <KPI title="Latency" value={typeof meta.latency === "number" ? `${meta.latency}ms` : "—"} />
              </Stack>

              <Divider my="md" />

              <Tabs defaultValue="sources" variant="pills">
                <Tabs.List>
                  <Tabs.Tab value="sources">Sources</Tabs.Tab>
                  <Tabs.Tab value="notes">Notes</Tabs.Tab>
                </Tabs.List>

                <Tabs.Panel value="sources" pt="sm">
                  <Stack gap="sm">
                    {meta.sources?.length ? (
                      meta.sources.slice(0, 4).map((s, i) => (
                        <Card key={i} radius="md" withBorder p="sm">
                          <Text fw={700} size="sm">
                            {s.title ?? `Source ${i + 1}`}
                          </Text>
                          {s.snippet && (
                            <Text size="sm" c="dimmed" lineClamp={3} mt={4}>
                              {s.snippet}
                            </Text>
                          )}
                          {s.url && (
                            <Button
                              component="a"
                              href={s.url}
                              target="_blank"
                              rel="noreferrer"
                              variant="light"
                              size="xs"
                              mt="sm"
                            >
                              Open
                            </Button>
                          )}
                        </Card>
                      ))
                    ) : (
                      <Text size="sm" c="dimmed">
                        No sources yet. (Backend can return `sources` to show citations.)
                      </Text>
                    )}
                  </Stack>
                </Tabs.Panel>

                <Tabs.Panel value="notes" pt="sm">
                  <Text size="sm" c="dimmed">
                    Suggested improvements:
                  </Text>
                  <Stack gap={6} mt="sm">
                    <Text size="sm">• Streaming tokens (SSE)</Text>
                    <Text size="sm">• “Copy answer” + feedback buttons</Text>
                    <Text size="sm">• Session memory + export transcript</Text>
                  </Stack>
                </Tabs.Panel>
              </Tabs>
            </Paper>
          </Stack>
        </AppShell.Aside>
      </AppShell>
    </Box>
  );
}

// Simple Markdown renderer with table support
function MarkdownContent({ content }: { content: string }) {
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let tableRows: string[][] = [];
  let inTable = false;
  let tableKey = 0;

  const processLine = (line: string, idx: number) => {
    // Table row
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      const cells = line.split('|').slice(1, -1).map(c => c.trim());
      // Skip separator rows (|---|---|)
      if (cells.every(c => /^[-:]+$/.test(c))) {
        return null;
      }
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      tableRows.push(cells);
      return null;
    } else if (inTable) {
      // End of table
      inTable = false;
      const table = (
        <Box key={`table-${tableKey++}`} style={{ overflowX: 'auto', margin: '8px 0' }}>
          <table style={{ 
            borderCollapse: 'collapse', 
            width: '100%', 
            fontSize: '13px',
            background: 'rgba(255,255,255,0.5)',
            borderRadius: '8px',
            overflow: 'hidden'
          }}>
            <thead>
              <tr style={{ background: 'rgba(99,102,241,0.1)' }}>
                {tableRows[0]?.map((cell, i) => (
                  <th key={i} style={{ 
                    padding: '8px 12px', 
                    textAlign: 'left', 
                    fontWeight: 600,
                    borderBottom: '2px solid rgba(99,102,241,0.2)'
                  }}>
                    {cell.replace(/\*\*/g, '')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.slice(1).map((row, ri) => (
                <tr key={ri} style={{ borderBottom: '1px solid rgba(0,0,0,0.05)' }}>
                  {row.map((cell, ci) => (
                    <td key={ci} style={{ padding: '8px 12px' }}>
                      {cell.replace(/\*\*/g, '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Box>
      );
      tableRows = [];
      elements.push(table);
    }

    // Process bold text into React elements
    const renderWithBold = (text: string) => {
      const parts = text.split(/(\*\*[^*]+\*\*)/g);
      return parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        return part;
      });
    };
    
    // List items
    if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      return (
        <div key={idx} style={{ paddingLeft: '16px', margin: '2px 0' }}>
          • {renderWithBold(line.trim().slice(2))}
        </div>
      );
    }
    
    // Numbered list
    const numMatch = line.trim().match(/^(\d+)\.\s+(.+)/);
    if (numMatch) {
      return (
        <div key={idx} style={{ paddingLeft: '16px', margin: '2px 0' }}>
          {numMatch[1]}. {renderWithBold(numMatch[2])}
        </div>
      );
    }

    // Checkbox items
    if (line.includes('- [ ]') || line.includes('- [x]')) {
      const checked = line.includes('- [x]');
      const text = line.replace(/- \[[ x]\]\s*/, '');
      return (
        <div key={idx} style={{ paddingLeft: '16px', margin: '2px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ opacity: 0.5 }}>{checked ? '☑' : '☐'}</span>
          <span>{renderWithBold(text)}</span>
        </div>
      );
    }

    // Regular line with bold support
    return line ? <div key={idx} style={{ margin: '2px 0' }}>{renderWithBold(line)}</div> : <div key={idx} style={{ height: '8px' }} />;
  };

  lines.forEach((line, idx) => {
    const el = processLine(line, idx);
    if (el) elements.push(el);
  });

  // Flush remaining table
  if (inTable && tableRows.length > 0) {
    elements.push(
      <Box key={`table-${tableKey}`} style={{ overflowX: 'auto', margin: '8px 0' }}>
        <table style={{ 
          borderCollapse: 'collapse', 
          width: '100%', 
          fontSize: '13px',
          background: 'rgba(255,255,255,0.5)',
          borderRadius: '8px'
        }}>
          <thead>
            <tr style={{ background: 'rgba(99,102,241,0.1)' }}>
              {tableRows[0]?.map((cell, i) => (
                <th key={i} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>
                  {cell.replace(/\*\*/g, '')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tableRows.slice(1).map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) => (
                  <td key={ci} style={{ padding: '8px 12px', borderBottom: '1px solid rgba(0,0,0,0.05)' }}>
                    {cell.replace(/\*\*/g, '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </Box>
    );
  }

  return <>{elements}</>;
}

function MessageBubble({ msg, reduceMotion }: { msg: Message; reduceMotion: boolean }) {
  const isUser = msg.role === "user";

  return (
    <motion.div
      initial={reduceMotion ? undefined : { opacity: 0, y: 10 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      exit={reduceMotion ? undefined : { opacity: 0, y: 6 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start" }}
    >
      <Group align="flex-start" gap="sm" wrap="nowrap" style={{ maxWidth: "85%" }}>
        {!isUser && (
          <ThemeIcon radius="xl" variant="filled" color="dark" size={34}>
            <IconShieldCheck size={16} />
          </ThemeIcon>
        )}

        <MotionCard
          radius="lg"
          withBorder
          p="sm"
          style={{
            background: isUser ? "#111827" : "#F4F6FF",
            color: isUser ? "white" : "#111827",
            borderColor: isUser ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.06)",
          }}
        >
          {isUser ? (
            <Text size="sm" style={{ whiteSpace: "pre-wrap", lineHeight: 1.55 }}>
              {msg.content}
            </Text>
          ) : (
            <Box style={{ fontSize: '14px', lineHeight: 1.55 }}>
              <MarkdownContent content={msg.content} />
            </Box>
          )}
        </MotionCard>

        {isUser && (
          <ThemeIcon radius="xl" variant="light" size={34}>
            <IconUser size={16} />
          </ThemeIcon>
        )}
      </Group>
    </motion.div>
  );
}

function KPI({ title, value, color }: { title: string; value: string; color?: string }) {
  return (
    <Paper radius="md" p="sm" withBorder>
      <Group justify="space-between">
        <Text size="sm" c="dimmed">
          {title}
        </Text>
        <Badge variant="light" color={color as any}>
          {value}
        </Badge>
      </Group>
    </Paper>
  );
}
