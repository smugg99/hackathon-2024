"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import remarkGfm from "remark-gfm";
import fs from "fs";
import path from "path";

const markdownFiles = [
  { title: "Depression", filename: "depression.md" },
  { title: "Anxiety", filename: "anxiety.md" },
  { title: "Stress Management", filename: "stress_management.md" },
];

async function getMarkdownContent(filename : string) {
  const filePath = path.join(process.cwd(), "data", filename);
  const markdownContent = fs.readFileSync(filePath, "utf8");
  return markdownContent;
}

const Tabs = () => {
  const [activeTab, setActiveTab] = useState(markdownFiles[0].filename);
  const [markdownContent, setMarkdownContent] = useState("");

  useEffect(() => {
    const loadContent = async () => {
      const content = await getMarkdownContent(activeTab);
      setMarkdownContent(content);
    };
    loadContent();
  }, [activeTab]);

  return (
    <div>
      <div className="flex border-b">
        {markdownFiles.map((file) => (
          <button
            key={file.filename}
            className={`px-4 py-2 text-sm font-semibold ${
              activeTab === file.filename ? "border-b-2 border-blue-500" : ""
            }`}
            onClick={() => setActiveTab(file.filename)}
          >
            {file.title}
          </button>
        ))}
      </div>
      <Card className="mt-4">
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
  );
};

export default Tabs;
