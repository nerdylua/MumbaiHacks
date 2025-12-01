"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Message } from "ai";
import { memo } from "react";
import { saveMessages } from "../actions";
interface SuggestedActionsProps {
  chatId: string;
  append: (message: Message) => Promise<string | null | undefined>;
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
}
function PureSuggestedActions({
  chatId,
  append,
  handleSubmit,
}: SuggestedActionsProps) {
  const suggestedActions = [
    {
      title: "Summarize Conference Call Transcript",
      label: "Summarize the conference call transcript and tell me why I should invest in the company",
      action:
        `Find me companies with strong fundamentals and high financial throughput, then choose the best company and get me their conference call transcript and summarize it, telling me why I should invest in them.`,
    },
    {
      title: "Search on Flipkart",
      label: "Search on Flipkart and show the results",
      action:
        `Find me mobile phones on Flipkart that are under 50,000/-. Then open them on the browser and show it to me`,
    },
    {
      title: "Identify Coverage Gaps",
      label: "Answer the question based on the document",
      action:
        `Answer based on this document, https://micbdyubdfqefphlaouz.supabase.co/storage/v1/object/public/documents/OrientalInsure.pdf "A 55-year-old on Silver with ₹5 lakh SI stays 2 days in ICU and 2 days in ward and uses an ambulance once. What is the ICU per-day admissible limit and how much ambulance expense is payable in this illness and policy period?"`,
    },
    {
      title: "Generate Chart",
      label: "Show me the charts for the companies",
      action:
        `Find me 3 companies with the best Price-to-Earnings (P/E) ratio and Return on Equity (ROE), then show me the charts for them.`,
    },
  ];
  return (
    <div className="grid sm:grid-cols-2 gap-2 w-full pb-2">
      {suggestedActions.map((suggestedAction, index) => (
        <motion.div
          initial={{
            opacity: 0,
            y: 20,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          exit={{
            opacity: 0,
            y: 20,
          }}
          transition={{
            delay: 0.05 * index,
          }}
          key={`suggested-action-${suggestedAction.title}-${index}`}
          className={index > 1 ? "hidden sm:block" : "block"}
        >
          <Button
            variant="ghost"
            onClick={async () => {
              const userMessage: Message = {
                id: crypto.randomUUID(),
                role: 'user',
                content: suggestedAction.action,
              };
              await saveMessages([userMessage], chatId);
              await append(userMessage);
            }}
            className="text-left border rounded-xl px-4 py-3.5 text-sm gap-1 sm:flex-col w-full h-auto justify-start items-start sm:items-stretch"
          >
            <span className="font-medium truncate">{suggestedAction.title}</span>
            <span className="text-muted-foreground truncate">
              {suggestedAction.label}
            </span>
          </Button>
        </motion.div>
      ))}
    </div>
  );
}
export const SuggestedActions = memo(PureSuggestedActions, () => true);
