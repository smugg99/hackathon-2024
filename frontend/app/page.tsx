import Link from "next/link";
import { CircleUser, Menu, Package2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import fs from "fs";
import path from "path";
import remarkGfm from "remark-gfm";

async function getMarkdownContent() {
  const filePath = path.join(process.cwd(), "data", "index.md");
  const markdownContent = fs.readFileSync(filePath, "utf8");
  return markdownContent;
}

export default async function Home() {
  const markdownContent = await getMarkdownContent();

  return (
    <div className="flex min-h-screen w-full flex-col">
      <header className="sticky top-0 flex h-16 items-center gap-4 border-b bg-background px-4 md:px-6">
        <nav className="hidden flex-col gap-6 text-lg font-medium md:flex md:flex-row md:items-center md:gap-5 md:text-sm lg:gap-6">
          <Link
            href="#"
            className="flex items-center gap-2 text-lg font-semibold md:text-base"
          >
            <Package2 className="h-6 w-6" />
            <span className="sr-only">Acme Inc</span>
          </Link>
          Balls
        </nav>
      </header>
      <main className="flex min-h-[calc(100vh_-_theme(spacing.16))] flex-1 flex-col gap-4 bg-muted/40 p-4 md:gap-8 md:p-10 h-full">
        <div className="mx-auto grid w-full h-full">
          <Card className="mx-auto lex-grow">
            <CardContent className="flex justify-center">
              <div className="prose text-left w-full max-w-3xl">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  components={{
                    h1: ({ children }) => <h1 className="text-4xl font-bold my-4">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-3xl font-semibold my-3">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-2xl font-medium my-2">{children}</h3>,
                    p: ({ children }) => <p className="my-2 text-lg">{children}</p>,
                    a: ({ children, href }) => (
                      <a href={href} className="text-blue-500 hover:underline">{children}</a>
                    ),
                    img: ({ alt, src }) => <img alt={alt} src={src} className="my-2 max-w-full rounded" />,
                    ul: ({ children }) => <ul className="list-disc list-inside my-2">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal list-inside my-2">{children}</ol>,
                    li: ({ children }) => <li className="ml-4 my-1">{children}</li>,
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-gray-300 pl-4 italic my-4">
                        {children}
                      </blockquote>
                    ),
                  }}
                >
                  {markdownContent}
                </ReactMarkdown>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
